from transformers import BertForQuestionAnswering

def initiate_model(remove_encoder, freeze):
  if remove_encoder:
    print('The last two BERT encoder layers are removed')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased', num_hidden_layers = 10)

  else:
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

  if freeze == True:
    print('All the embedding parameters in BERT model are frozen.')
    for name, param in model.named_parameters():
      if 'bert.embeddings' in name:
        param.requires_grad = False

  return model

remove_encoder = False
freeze = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fb_model = initiate_model(remove_encoder, freeze).to(device)
