from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
import torch
from torch.nn import CrossEntropyLoss

from .quant_layer import encoder

class QuantRoberta:
    def __init__(self):
        self.model = RobertaForSequenceClassification.from_pretrained('textattack/roberta-base-MRPC')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def eval(self):
        self.model.eval()

    def sequence_classification(self,
                                outputs, 
                                input_ids=None,
                                attention_mask=None,
                                token_type_ids=None,
                                position_ids=None,
                                head_mask=None,
                                inputs_embeds=None,
                                labels=None,
                                output_attentions=None,
                                output_hidden_states=None,
                                return_dict=None,):
        sequence_output = outputs[0]
        logits = self.model.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.model.config.problem_type is None:
                if self.model.num_labels == 1:
                    self.model.config.problem_type = "regression"
                elif self.model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.model.config.problem_type = "single_label_classification"
                else:
                    self.model.config.problem_type = "multi_label_classification"

            if self.model.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.model.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
            elif self.model.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def __call__(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        embedding_output = self.model.roberta.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
        )
        extended_attention_mask = self.model.roberta.get_extended_attention_mask(attention_mask, input_ids.size(), self.device)
        hidden_states = encoder(self.model.roberta, embedding_output, extended_attention_mask)

        pooled_output = self.model.roberta.pooler(hidden_states) if self.model.roberta.pooler is not None else None
        
        outputs = BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=hidden_states,
                pooler_output=pooled_output,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
            )

        outputs = self.sequence_classification(outputs, labels=labels)

        return outputs

