from transformers import BertModel
from gat import SimpleGAT
from TTransformer import TTransformerModel
from bert_gat import SimpleGAT_BERT
# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 64, 2

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freeze the BERT model
        #if freeze_bert:
        #    for param in self.bert.parameters():
        #        param.requires_grad = False

    def forward(self, data):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """

        input_ids, attention_mask = data.input_ids, data.attention_mask
        print('input_ids shape', input_ids.shape)
        print('attention_mask shape', attention_mask.shape)
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        #logits = self.classifier(last_hidden_state_cls)

        return last_hidden_state_cls




class ComboNet(nn.Module):
    def __init__(self,user_in, user_hid, user_out, in_feats,hid_feats,out_feats,D_in, D_H, D_out):
        super(ComboNet, self).__init__()
        #D_in, H, D_out = 768,64,4
        #self.bert_seq = BertClassifier(freeze_bert=False)
        self.bert_tt = TTransformerModel()
        self.user_gat = SimpleGAT(user_in, user_hid, user_out)
        self.bert_gat = SimpleGAT_BERT(in_feats,hid_feats,out_feats)
        #self.gnn = SimpleTDrumorGCN_ROOT(in_feats, hid_feats, out_feats)
        self.fc1 = nn.Linear((out_feats+user_out+D_in),H)
        self.fc2 = nn.Linear(H,D_out)

    def forward(self,data):
        #bert_x = self.bert(data)
        seq_x = self.bert_tt(data)
        user_x = self.user_gat(data)
        bert_gat_x = self.bert_gat(data)

        x = torch.cat((bert_gat_x,user_x,seq_x), 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x= F.log_softmax(x, dim=1)
        return x
