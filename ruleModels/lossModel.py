from model import *

class LossModel(HALOModel):
    def __init__(self, config):
        super(HALOModel, self).__init__()
        self.transformer = CoarseTransformerModel(config)
        self.ehr_head = FineAutoregressiveHead(config)
        self.rules = [r for r in config.rules if r[0] != -1 and len(r[0]) <= 1]
        self.n_ctx = config.n_ctx
        self.total_vocab_size = config.total_vocab_size
        self.w = config.semantic_w

    def forward(self, input_visits, position_ids=None, ehr_labels=None, ehr_masks=None, past=None):
        hidden_states = self.transformer(input_visits, position_ids, past)
        code_logits = self.ehr_head(hidden_states, input_visits)
        sig = nn.Sigmoid()
        code_probs = sig(code_logits)
        if ehr_labels is not None:    
            shift_labels = ehr_labels[..., 1:, :].contiguous()
            if ehr_masks is not None:
                code_probs = code_probs * ehr_masks
                shift_labels = shift_labels * ehr_masks

            bce = nn.BCELoss()
            classification_loss = bce(code_probs, shift_labels)
            semantic_loss = 0
            for (past_visits, past_codes, past_non_codes, current_codes, current_non_codes, output_code, output_value) in self.rules:
                pv = past_visits[0] if len(past_visits) > 0 else -1
                pos_prob = code_probs[:,:,output_code] if output_value == 1 else 1 - code_probs[:,:,output_code]
                neg_prob = 1-pos_prob
                prob_list = torch.zeros(code_probs.size(0),code_probs.size(1),len(past_codes)+len(past_non_codes)+len(current_codes)+len(current_non_codes), dtype=torch.float32, device=code_probs.device)
                currCounter = 0
                
                if pv >= 0:
                    for c in past_codes:

                        prob_list[:,:,currCounter] = code_probs[:,pv,c].unsqueeze(1).repeat(1,prob_list.size(1))
                        currCounter += 1
                    
                    for c in past_non_codes:
                        prob_list[:,:,currCounter] = 1 - code_probs[:,pv,c].unsqueeze(1).repeat(1,prob_list.size(1))
                        currCounter += 1
                else:
                    for c in past_codes:
                        prob_list[:,:,currCounter] = code_probs[list(range(code_probs.size(0))), [0 if i+pv < 0  else i+pv for i in range(code_probs.size(1))], c]
                        currCounter += 1
                    
                    for c in past_non_codes:
                        prob_list[:,:,currCounter] = 1 - code_probs[list(range(code_probs.size(0))), [0 if i+pv < 0  else i+pv for i in range(code_probs.size(1))], c]
                        currCounter += 1
                
                for c in current_codes:
                    prob_list[:,:,currCounter] = code_probs[:,:,c]
                    currCounter += 1
                    
                for c in current_non_codes:
                    prob_list[:,:,currCounter] = 1 - code_probs[:,:,c]
                    currCounter += 1

                satis_prob = prob_list.prod(dim=-1)
                nonsatis_prob = 1-satis_prob
                rule_prob = -torch.log((satis_prob*pos_prob) + (nonsatis_prob*neg_prob))   
                semantic_loss += rule_prob.mean()
                
            semantic_loss = semantic_loss/len(self.rules)
            loss = classification_loss + self.w * semantic_loss
            return loss, code_probs, shift_labels
        
        return code_probs

    def sample(self, input_visits, random=True):
        sig = nn.Sigmoid()
        hidden_states = self.transformer(input_visits)
        i = 0
        while i < self.ehr_head.tot_vocab:
            next_logits = self.ehr_head.sample(hidden_states, input_visits)
            next_probs = sig(next_logits)
            if random:
                visit = torch.bernoulli(next_probs)
            else:
                visit = torch.round(next_probs)
            
            remaining_visit = visit[:,0,i:]
            nonzero = torch.nonzero(remaining_visit, as_tuple=True)[1]
            if nonzero.numel() == 0:
                break

            first_nonzero = nonzero.min()
            input_visits[:,-1,i + first_nonzero] = visit[:,0,i + first_nonzero]
            i = i + first_nonzero + 1
    
        return input_visits