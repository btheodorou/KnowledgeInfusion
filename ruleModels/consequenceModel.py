from model import *

class ConSequenceModel(HALOModel):
    def __init__(self, config):
        super(HALOModel, self).__init__()
        self.transformer = CoarseTransformerModel(config)
        self.ehr_head = FineAutoregressiveHead(config)
        rulesPV = []
        rulesRN = []
        rulesRT = []
        rulesRO = []
        currPast = None
        currOutputs = set()
        for (past_visits, past_codes, past_non_codes, current_codes, current_non_codes, output_code, output_value) in config.rules:
            if currPast != past_visits or any([c in currOutputs for c in current_codes + current_non_codes + [output_code]]):
                # Add the last group of rules
                if currPast is not None:
                    rulesPV.append(nn.Parameter(pastVisits, requires_grad=False)) 
                    rulesRN.append(nn.Parameter(ruleNeuron, requires_grad=False)) 
                    rulesRT.append(nn.Parameter(ruleThreshold, requires_grad=False))
                    rulesRO.append(nn.Parameter(ruleOutput, requires_grad=False))
                    
                # Reset the group separators
                currPast = past_visits
                currOutputs = set()
                
                # Create the new temporal aggregation matrix
                if past_visits == -1:
                    pastVisits = torch.tril(torch.ones(config.n_ctx, config.n_ctx), diagonal=-1)
                else:
                    pastVisits = torch.zeros(config.n_ctx, config.n_ctx)
                    for v in past_visits:
                        if v >= 0:
                            pastVisits[v+1:,v] = 1
                        else:
                            pastVisits[list(range(-v, config.n_ctx)), [i + v for i in range(-v, config.n_ctx)]]
                
                # Create the new rule neuron
                ruleNeuron = torch.zeros(2*config.total_vocab_size, config.total_vocab_size)
                ruleThreshold = -1 * torch.ones(config.total_vocab_size)
                ruleOutput = torch.zeros(config.total_vocab_size)
            
            # Add the current rule to the rule neuron
            ruleNeuron[[config.total_vocab_size+c for c in past_non_codes], output_code] = -1
            ruleNeuron[[config.total_vocab_size+c for c in past_codes], output_code] = 1
            ruleNeuron[current_non_codes, output_code] = -1
            ruleNeuron[current_codes, output_code] = 1
            ruleThreshold[output_code] = len(current_codes) + len(past_codes)
            ruleOutput[output_code] = output_value
            
            # Add the current output to the set of outputs
            currOutputs.add(output_code)
                
        rulesPV.append(nn.Parameter(pastVisits, requires_grad=False)) 
        rulesRN.append(nn.Parameter(ruleNeuron, requires_grad=False)) 
        rulesRT.append(nn.Parameter(ruleThreshold, requires_grad=False))
        rulesRO.append(nn.Parameter(ruleOutput, requires_grad=False))
        self.rulesPV = nn.ParameterList(rulesPV)
        self.rulesRN = nn.ParameterList(rulesRN)
        self.rulesRT = nn.ParameterList(rulesRT)
        self.rulesRO = nn.ParameterList(rulesRO)
        
    def reset_rules(self, config):
        rulesPV = []
        rulesRN = []
        rulesRT = []
        rulesRO = []
        currPast = None
        currOutputs = set()
        for (past_visits, past_codes, past_non_codes, current_codes, current_non_codes, output_code, output_value) in config.rules:
            if currPast != past_visits or any([c in currOutputs for c in current_codes + current_non_codes + [output_code]]):
                # Add the last group of rules
                if currPast is not None:
                    rulesPV.append(nn.Parameter(pastVisits, requires_grad=False)) 
                    rulesRN.append(nn.Parameter(ruleNeuron, requires_grad=False)) 
                    rulesRT.append(nn.Parameter(ruleThreshold, requires_grad=False))
                    rulesRO.append(nn.Parameter(ruleOutput, requires_grad=False))
                    
                # Reset the group separators
                currPast = past_visits
                currOutputs = set()
                
                # Create the new temporal aggregation matrix
                if past_visits == -1:
                    pastVisits = torch.tril(torch.ones(config.n_ctx, config.n_ctx), diagonal=-1)
                else:
                    pastVisits = torch.zeros(config.n_ctx, config.n_ctx)
                    for v in past_visits:
                        if v >= 0:
                            pastVisits[v+1:,v] = 1
                        else:
                            pastVisits[list(range(-v, config.n_ctx)), [i + v for i in range(-v, config.n_ctx)]]
                
                # Create the new rule neuron
                ruleNeuron = torch.zeros(2*config.total_vocab_size, config.total_vocab_size)
                ruleThreshold = -1 * torch.ones(config.total_vocab_size)
                ruleOutput = torch.zeros(config.total_vocab_size)
            
            # Add the current rule to the rule neuron
            ruleNeuron[[config.total_vocab_size+c for c in past_non_codes], output_code] = -1
            ruleNeuron[[config.total_vocab_size+c for c in past_codes], output_code] = 1
            ruleNeuron[current_non_codes, output_code] = -1
            ruleNeuron[current_codes, output_code] = 1
            ruleThreshold[output_code] = len(current_codes) + len(past_codes)
            ruleOutput[output_code] = output_value
            
            # Add the current output to the set of outputs
            currOutputs.add(output_code)
                
        rulesPV.append(nn.Parameter(pastVisits, requires_grad=False)) 
        rulesRN.append(nn.Parameter(ruleNeuron, requires_grad=False)) 
        rulesRT.append(nn.Parameter(ruleThreshold, requires_grad=False))
        rulesRO.append(nn.Parameter(ruleOutput, requires_grad=False))
        self.rulesPV = nn.ParameterList(rulesPV)
        self.rulesRN = nn.ParameterList(rulesRN)
        self.rulesRT = nn.ParameterList(rulesRT)
        self.rulesRO = nn.ParameterList(rulesRO)

    def forward(self, input_visits, position_ids=None, ehr_labels=None, ehr_masks=None, past=None):
        hidden_states = self.transformer(input_visits, position_ids, past)
        code_logits = self.ehr_head(hidden_states, input_visits)
        sig = nn.Sigmoid()
        code_probs = sig(code_logits)
        for i in range(len(self.rulesRN)):
            pastVisits = self.rulesPV[i]
            ruleNeuron = self.rulesRN[i]
            ruleThreshold = self.rulesRT[i]
            ruleOutput = self.rulesRO[i]
            past_visits = (pastVisits @ input_visits).bool().float()
            neuron_input = torch.cat((input_visits, past_visits), dim=-1)[:,1:]
            code_probs = torch.where((neuron_input @ ruleNeuron) == ruleThreshold, ruleOutput, code_probs)
        if ehr_labels is not None:    
            shift_labels = ehr_labels[..., 1:, :].contiguous()
            if ehr_masks is not None:
                code_probs = code_probs * ehr_masks
                shift_labels = shift_labels * ehr_masks

            bce = nn.BCELoss()
            loss = bce(code_probs, shift_labels)
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
         
        for i in range(len(self.rulesRN)):
            pastVisits = self.rulesPV[i]
            ruleNeuron = self.rulesRN[i]
            ruleThreshold = self.rulesRT[i]
            ruleOutput = self.rulesRO[i]
            past_visits = (pastVisits[:input_visits.size(1), :input_visits.size(1)] @ input_visits).bool().float()
            neuron_input = torch.cat((input_visits, past_visits), dim=-1)
            input_visits = torch.where((neuron_input @ ruleNeuron) == ruleThreshold, ruleOutput, input_visits)

        return input_visits