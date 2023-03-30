from model import *

class CCNModel(HALOModel):
    def __init__(self, config):
        super(HALOModel, self).__init__()
        self.transformer = CoarseTransformerModel(config)
        self.ehr_head = FineAutoregressiveHead(config)
        
        Iplus = []
        Iminus = []
        M = []
        for (past_visits, past_codes, past_non_codes, current_codes, current_non_codes, output_code, output_value) in config.rules:
            if past_visits != [] or output_value == 0:
                continue
            
            iplus = torch.zeros(config.total_vocab_size)
            iminus = torch.zeros(config.total_vocab_size)
            m = torch.zeros(config.total_vocab_size)
            for c in current_codes:
                iplus[c] = 1
            for c in current_non_codes:
                iminus[c] = 1
            m[output_code] = 1
            
            Iplus.append(iplus)
            Iminus.append(iminus)
            M.append(m)
        self.Iplus = nn.Parameter(torch.stack(Iplus), requires_grad=False)
        self.Iminus = nn.Parameter(torch.stack(Iminus), requires_grad=False)
        self.M = nn.Parameter(torch.stack(M).transpose(0,1), requires_grad=False)
        
    # Get the vector containing the value of each body having more than one literal
    def get_v(self, out):
        out = out.unsqueeze(1)
        H = out.expand(len(out), len(self.Iplus), len(self.Iplus[0]))
        Iplus, Iminus = self.Iplus.unsqueeze(0), self.Iminus.unsqueeze(0)
        Iplus = Iplus.expand(len(out),len(Iplus[0]),len(Iplus[0][0]))
        Iminus = Iminus.expand(len(out),len(Iplus[0]),len(Iplus[0][0]))
        v, _ = torch.min((((Iplus*H)+(Iminus*(1-H)))+(1-Iplus-Iminus))*(1-Iplus*Iminus)+torch.min((Iplus*H),Iminus*(1-H))*Iplus*Iminus,dim=2)
        return v

    # Get the value of the constrained output
    def get_constr_out(self, x, device):
        bs, ts, vs = x.size(0), x.size(1), x.size(2)
        x = x.view(-1, vs)
        c_out = x
        v = self.get_v(c_out)
        
        # Concatenate the output of the network with the "value" associated to each body having more than one literal
        # and then expand it to a tensor [batch_size, num_classes, num_rules] <-- num rules stnads for the number of rules having len(body) > 1
        V = torch.cat((c_out,v), dim=1)
        V = V.unsqueeze(1)
        V = V.expand(len(x),len(self.Iplus[0]), len(self.Iplus[0])+len(v[0]))
        
        # Concatenate the matrix encoding the hierarchy (i.e., one literal rules) with the matrix M
        # which encodes which body corresponds to which head
        M = self.M.unsqueeze(0)
        R = torch.eye(len(self.Iplus[0])).unsqueeze(0).to(device)
        R = torch.cat((R,M),dim=2)
        R_batch = R.expand(len(x),len(self.Iplus[0]), len(self.Iplus[0])+len(v[0]))

        #Compute the final output
        final_out, _ = torch.max(R_batch*V, dim = 2)
        final_out = final_out.view(bs, ts, vs)
        return final_out

    # Get the vector containing the value of each implicant having more than one literal for the training phase
    def get_v_train(self, out, y, label_polarity):
        # H has shape (batch_size, num_rules, num_classes) <-- num rules stands for the number of rules having len(body) > 1
        out = out.unsqueeze(1)
        H = out.expand(len(out),len(self.Iplus),len(self.Iplus[0]))
        
        y = y.unsqueeze(1)
        Y = y.expand(len(y),len(self.Iplus),len(self.Iplus[0]))

        Iplus, Iminus = self.Iplus.unsqueeze(0), self.Iminus.unsqueeze(0)
        Iplus = Iplus.expand(len(out),len(Iplus[0]),len(Iplus[0][0]))
        Iminus = Iminus.expand(len(out),len(Iplus[0]),len(Iplus[0][0]))
        
        if label_polarity=='positive':
            vplus, _ = torch.min(Iplus*H*Y+(1-Iplus),dim=2)
            vminus,_ = torch.min(Iminus*(1-H)*(1-Y)+(1-Iminus),dim=2)
            v = torch.min(vplus,vminus)
        else:
            vplus, _ = torch.min(Iplus*H*(1-Y)+(1-Iplus)+Iplus*Y, dim=2)
            vminus,_ = torch.min(Iminus*(1-H)*Y+(1-Iminus)+Iminus*(1-Y), dim=2)
            v = torch.min(vplus,vminus)
        
        return v

    #Get the value of the constrained output
    def get_constr_out_train(self, x, y, device, label_polarity):
        assert(label_polarity=='positive' or label_polarity=='negative')
        bs, ts, vs = x.size(0), x.size(1), x.size(2)
        new_bs = bs*ts
        x = x.view(new_bs, vs)
        y = y.view(new_bs, vs)
        
        out = x
        v = self.get_v_train(out, y, label_polarity)

        y = y.unsqueeze(1)
        Y = y.expand(len(y),len(self.Iplus),len(self.Iplus[0]))   
        
        # Concatenate the output of the network with the "value" of each implicant having more than one literal
        # and then expand it to a tensor [batch_size, num_classes, num_rules]
        V = torch.cat((out,v), dim=1)
        V = V.unsqueeze(1)
        V = V.expand(len(x),len(self.Iplus[0]), len(self.Iplus[0])+len(v[0]))
        
        # Concatenate the matrix encoding the hierarchy (i.e., one literal rules) with the matrix M
        # which encodes which body corresponds to which head
        M = self.M.unsqueeze(0) 
        R = torch.eye(len(self.Iplus[0])).unsqueeze(0).to(device)
        R = torch.cat((R,M),dim=2)
        R = R.expand(len(x),len(self.Iplus[0]), len(self.Iplus[0])+len(v[0]))
        # Compute the final output
        final_out, _ = torch.max(R*V, dim = 2)
        
        final_out = final_out.view(bs, ts, vs)
        return final_out

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
            if self.training:
                train_output_plus = self.get_constr_out_train(code_probs, shift_labels, code_probs.device, label_polarity='positive',)
                train_output_minus = self.get_constr_out_train(code_probs, shift_labels, code_probs.device, label_polarity='negative',)
                train_output = (train_output_plus*shift_labels)+(train_output_minus*(1-shift_labels))    
                loss = bce(train_output, shift_labels)
                return loss, train_output, shift_labels
            else: 
                code_probs = self.get_constr_out(code_probs, code_probs.device)
                loss = bce(code_probs, shift_labels)
                return loss, code_probs, shift_labels
        
        code_probs = self.get_constr_out(code_probs, code_probs.device)
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
            
        input_visits = self.get_constr_out(input_visits, input_visits.device)
        return input_visits