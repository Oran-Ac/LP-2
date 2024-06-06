from absl import app
from absl import flags
from tqdm.auto import tqdm
from transformers import AutoTokenizer,get_cosine_schedule_with_warmup,AutoModelForSequenceClassification
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed,DistributedDataParallelKwargs
import json
import random
import numpy as np
import torch
import torch.nn as nn
import logging
import datetime
import math
import os
# import self-defined functions
from dataset import *


logger = get_logger('my_logger')

FLAGS = flags.FLAGS
# data
flags.DEFINE_string('data_dict',None,'The directory of the data')
flags.DEFINE_integer('max_length',256,'The maximum length of the input')
flags.DEFINE_string('category','easy','The category of the data')
# model
flags.DEFINE_string('model_type','bert-base-uncased','The type of the model')
# training
flags.DEFINE_integer('batch_size',16,'The batch size')
flags.DEFINE_integer('num_epochs',3,'The number of epochs')
# optimizer
flags.DEFINE_integer('seed',42,'The seed for the random number generator')
flags.DEFINE_integer('patience',3,'patience')
flags.DEFINE_integer('gradient_accumulation_steps',1,'gradient accumulation steps')
flags.DEFINE_string('optimizer','sgd','optimizer')
flags.DEFINE_string('lr_decay_type','cosine','lr decay type')
flags.DEFINE_float('momentum',0.9,'momentum')
flags.DEFINE_float('lr',1e-5,'learning rate')
flags.DEFINE_float('weight_decay',1e-8,'weight decay')
flags.DEFINE_float('warmup_ratio',0.1,'warmup ratio')
# save
flags.DEFINE_string('output_model_dir','./saved_model','output directory')
flags.DEFINE_string('output_log_dir','./log','output directory')
# curriculum_learning learning
flags.DEFINE_boolean('curriculum_learning',True,'curriculum learning')
flags.DEFINE_boolean('distill',False,'distill')
flags.DEFINE_string('teacher_model_type','bert-base-uncased','The type of the teacher model')
flags.DEFINE_string('teacher_model_path',None,'The path of the teacher model')
flags.DEFINE_float('temperature',1.0,'The temperature for distillation')

def set_seed(seed):
    # set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def data4curriculum_learning(data_dict,tokenizer_type,max_length):
    # categories = ['easy','medium','hard']
    # categories = ['medium','hard']
    categories = ['hard']
    # categories = ['medium']
    logger.info('Start preparing the data for curriculum learning')
    for c in categories:
        logger.info(f'Preparing the data for {c}')
        train_dataset = ClassificationDataset(data_dict,c,'train',tokenizer_type,max_length)
        validation_dataset = ClassificationDataset(data_dict,c,'validation',tokenizer_type,max_length)
        yield train_dataset,validation_dataset,c

def main(argv):
    # set seed
    set_seed(FLAGS.seed)
    # folder
    if not os.path.exists(FLAGS.output_model_dir):
        os.makedirs(FLAGS.output_model_dir)
    if not os.path.exists(FLAGS.output_log_dir):
        os.makedirs(FLAGS.output_log_dir)
    # define the accelerator
    accelerator = Accelerator(log_with="wandb")
    if accelerator.is_main_process:
        now_time = datetime.datetime.now().strftime('%m-%d_%H-%M%-S')
        # configure logging
        fileHeader = logging.FileHandler(
                filename = os.path.join(FLAGS.output_log_dir,f'{FLAGS.model_type.replace("/","-")}_{now_time}.log'), 
                mode = 'w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        fileHeader.setFormatter(formatter)
        # call inside the logger:https://github.com/huggingface/accelerate/blob/658492fb410396d8d1c1241c1cc2412a908b431b/src/accelerate/logging.py#L112
        logger.logger.addHandler(fileHeader)
        logger.logger.setLevel(logging.INFO)


        # logging the flags
        logger.info(json.dumps(FLAGS.flag_values_dict(),indent=4))
        logger.info(accelerator.state, main_process_only=False) 
    experiment_config = FLAGS.flag_values_dict()
    accelerator.init_trackers(FLAGS.model_type.replace("/","-"), experiment_config, init_kwargs={"wandb": {"mode": "offline",'config':experiment_config}})
    # load the model
    model = AutoModelForSequenceClassification.from_pretrained(FLAGS.model_type,num_labels=2)
    
    # load the data
    if FLAGS.curriculum_learning:
        for train_dataset,validation_dataset, category in data4curriculum_learning(FLAGS.data_dict,FLAGS.model_type,FLAGS.max_length):
            # data loader
            train_dataloader =  torch.utils.data.DataLoader(
                train_dataset,
                batch_size=FLAGS.batch_size,
                collate_fn=ClassificationsCollator(),
                shuffle=True
            )
            validation_dataloader = torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=FLAGS.batch_size,
                collate_fn=ClassificationsCollator(),
                shuffle=False
            )
            # define the optimizer
            if FLAGS.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(),lr=FLAGS.lr,momentum=FLAGS.momentum,weight_decay=FLAGS.weight_decay)
            elif FLAGS.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(),lr=FLAGS.lr,weight_decay=FLAGS.weight_decay)
            else:
                raise NotImplementedError('Not implemented yet')

            # define the lr scheduler
            num_training_steps = len(train_dataset) // FLAGS.batch_size * FLAGS.num_epochs
            num_warmup_steps = math.ceil(num_training_steps * FLAGS.warmup_ratio)
            if FLAGS.lr_decay_type == 'cosine':
                lr_scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps,num_training_steps)
            else:
                raise NotImplementedError('Not implemented yet')
            # define the loss function
            loss_fn = nn.CrossEntropyLoss()
            # Prepare everything with our `accelerator`.
            if FLAGS.distill:
                teacher = AutoModelForSequenceClassification.from_pretrained(FLAGS.teacher_model_type.replace("/","-"),num_labels=2)
                teacher.load_state_dict(torch.load(os.path.join(FLAGS.teacher_model_path,FLAGS.teacher_model_type.replace("/","-"),category+'.pt')))
                teacher = teacher.to(accelerator.device)
                teacher.eval()
                distill_fn = nn.KLDivLoss(reduction='batchmean')

            train_dataloader, validation_dataloader, model, optimizer = accelerator.prepare(
                train_dataloader, validation_dataloader, model, optimizer
            )
            # train
            best_acc = 0.0
            patience_counter = 0
            total_batch_size = FLAGS.batch_size * accelerator.num_processes * FLAGS.gradient_accumulation_steps
            if accelerator.is_main_process:
                logger.info("***** Running training *****")
                logger.info(f"  Num examples = {len(train_dataset)}")
                logger.info(f"  Num Epochs = {FLAGS.num_epochs}")
                logger.info(f"  Instantaneous batch size per device = {FLAGS.batch_size}")
                logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
                logger.info(f"  Gradient Accumulation steps = {FLAGS.gradient_accumulation_steps}")
                logger.info(f"  Total optimization steps = {num_training_steps}")
            progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_main_process)
            for epoch in range(FLAGS.num_epochs):
                total_loss = []
                distill_loss = []
                learning_loss = []
                model.train()
                for step, batch in enumerate(train_dataloader):
                    outputs = model(**batch)
                    loss = loss_fn(outputs.logits,batch['labels'])
                    if FLAGS.distill:
                        learning_loss.append(loss.item())
                        with torch.no_grad():
                            teacher_outputs = teacher(**batch)
                        distill_loss = distill_fn(outputs.logits / FLAGS.temperature,teacher_outputs.logits / FLAGS.temperature)
                        loss = (1.0 - FLAGS.temperature) * loss + FLAGS.temperature * distill_loss
                    total_loss.append(loss.item())
                    if FLAGS.distill:
                        distill_loss.append(distill_loss.item())
                    loss = loss / FLAGS.gradient_accumulation_steps
                    accelerator.backward(loss)
                    if step % FLAGS.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                logger.info(f'Epoch {epoch+1}/{FLAGS.num_epochs} - Training Loss: {np.mean(total_loss)}')
                accelerator.log(
                    {
                    "train/epoch": epoch,
                    "train/loss": np.mean(total_loss),
                    "train/learning_loss": np.mean(learning_loss),
                    "train/distill_loss": np.mean(distill_loss),
                }
                )
            
                # evaluate
                model.eval()
                all_preds = []
                all_labels = []
                total_loss = []
                for batch in validation_dataloader:
                    with torch.no_grad():
                        outputs = model(**batch)
                    all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())
                    loss = loss_fn(outputs.logits,batch['labels'])
                    total_loss.append(loss.item())
                acc = np.mean(np.array(all_preds) == np.array(all_labels))
                if accelerator.is_main_process:
                    logger.info(f'Epoch {epoch+1}/{FLAGS.num_epochs} - Validation Accuracy: {acc} - Validation Loss: {np.mean(total_loss)}')
                    accelerator.log(
                        {
                        "validation/epoch": epoch,
                        "validation/loss": np.mean(total_loss),
                        "validation/accuracy": acc,
                    }
                )
                if acc > best_acc:
                    best_acc = acc
                    patience_counter = 0
                    if accelerator.is_main_process:
                        model_path = os.path.join(FLAGS.output_model_dir,f'{FLAGS.model_type.replace("/","-")}_{FLAGS.category}.pt')
                        accelerator.save(model.state_dict(),model_path)
                        logger.info(f'Save the model at {model_path}')
                else:
                    patience_counter += 1
                    if patience_counter >= FLAGS.patience:
                        break
                
                


    else:
        raise NotImplementedError('Not implemented yet')

if __name__ == '__main__':
    app.run(main)
