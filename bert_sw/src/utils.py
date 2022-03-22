from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.data.metrics import simple_accuracy
import torch
from tqdm import tqdm
import numpy as np

def roberta_mrpc_dataset():

    def encode(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')

    dataset = load_dataset('glue', 'mrpc', split='validation')
    tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-MRPC')
    tokenizer.decode(tokenizer(dataset[0]['sentence1'], dataset[0]['sentence2'])['input_ids'])
    dataset = dataset.map(encode, batched=True)
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return dataset
    
def eval_model(model, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    preds = None

    for i, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():

            outputs = model(**batch)
            
            tmp_eval_loss, logits = outputs[:2]
            
            loss = outputs[0]
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = batch['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, batch['labels'].detach().cpu().numpy(), axis=0)
        if i % 10 == 0:
    #         print(f"loss: {loss}")
            pass

    preds = np.argmax(preds, axis=1)

    print(f'accuracy: {simple_accuracy(preds, out_label_ids)}')