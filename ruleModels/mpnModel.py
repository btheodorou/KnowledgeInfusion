from model import *
from symbolic import symbolic
from torch import nn
class MultiPlexNetModel(HALOModel):
    def __init__(self, config):
        super(HALOModel, self).__init__()
        self.transformer = CoarseTransformerModel(config)
        self.ehr_head = FineAutoregressiveHead(config)
        self.rules = config.rules
        self.n_ctx = config.n_ctx
        self.total_vocab_size = config.total_vocab_size
        dnf_rules = "(1158 & 1610 & 288 & 93 & ~1611 & ~1612 & ~1613 & ~1614) | (1158 & 1610 & 288 & 93 & ~1611 & ~1612 & ~1613 & ~1615) | (1158 & 1611 & 288 & 93 & ~1610 & ~1612 & ~1613 & ~1614) | (1158 & 1611 & 288 & 93 & ~1610 & ~1612 & ~1613 & ~1615) | (1158 & 1612 & 288 & 93 & ~1610 & ~1611 & ~1613 & ~1614) | (1158 & 1612 & 288 & 93 & ~1610 & ~1611 & ~1613 & ~1615) | (1158 & 1613 & 288 & 93 & ~1610 & ~1611 & ~1612 & ~1614) | (1158 & 1613 & 288 & 93 & ~1610 & ~1611 & ~1612 & ~1615) | (1158 & 1610 & 93 & ~1366 & ~1611 & ~1612 & ~1613 & ~1614) | (1158 & 1610 & 93 & ~1366 & ~1611 & ~1612 & ~1613 & ~1615) | (1158 & 1611 & 93 & ~1366 & ~1610 & ~1612 & ~1613 & ~1614) | (1158 & 1611 & 93 & ~1366 & ~1610 & ~1612 & ~1613 & ~1615) | (1158 & 1612 & 93 & ~1366 & ~1610 & ~1611 & ~1613 & ~1614) | (1158 & 1612 & 93 & ~1366 & ~1610 & ~1611 & ~1613 & ~1615) | (1158 & 1613 & 93 & ~1366 & ~1610 & ~1611 & ~1612 & ~1614) | (1158 & 1613 & 93 & ~1366 & ~1610 & ~1611 & ~1612 & ~1615) | (1610 & 288 & 93 & ~1297 & ~1611 & ~1612 & ~1613 & ~1614) | (1610 & 288 & 93 & ~1297 & ~1611 & ~1612 & ~1613 & ~1615) | (1611 & 288 & 93 & ~1297 & ~1610 & ~1612 & ~1613 & ~1614) | (1611 & 288 & 93 & ~1297 & ~1610 & ~1612 & ~1613 & ~1615) | (1612 & 288 & 93 & ~1297 & ~1610 & ~1611 & ~1613 & ~1614) | (1612 & 288 & 93 & ~1297 & ~1610 & ~1611 & ~1613 & ~1615) | (1613 & 288 & 93 & ~1297 & ~1610 & ~1611 & ~1612 & ~1614) | (1613 & 288 & 93 & ~1297 & ~1610 & ~1611 & ~1612 & ~1615) | (1610 & 93 & ~1297 & ~1366 & ~1611 & ~1612 & ~1613 & ~1614) | (1610 & 93 & ~1297 & ~1366 & ~1611 & ~1612 & ~1613 & ~1615) | (1611 & 93 & ~1297 & ~1366 & ~1610 & ~1612 & ~1613 & ~1614) | (1611 & 93 & ~1297 & ~1366 & ~1610 & ~1612 & ~1613 & ~1615) | (1612 & 93 & ~1297 & ~1366 & ~1610 & ~1611 & ~1613 & ~1614) | (1612 & 93 & ~1297 & ~1366 & ~1610 & ~1611 & ~1613 & ~1615) | (1613 & 93 & ~1297 & ~1366 & ~1610 & ~1611 & ~1612 & ~1614) | (1613 & 93 & ~1297 & ~1366 & ~1610 & ~1611 & ~1612 & ~1615) | (1158 & 1610 & 288 & ~1196 & ~1611 & ~1612 & ~1613 & ~1614 & ~514) | (1158 & 1610 & 288 & ~1196 & ~1611 & ~1612 & ~1613 & ~1615 & ~514) | (1158 & 1611 & 288 & ~1196 & ~1610 & ~1612 & ~1613 & ~1614 & ~514) | (1158 & 1611 & 288 & ~1196 & ~1610 & ~1612 & ~1613 & ~1615 & ~514) | (1158 & 1612 & 288 & ~1196 & ~1610 & ~1611 & ~1613 & ~1614 & ~514) | (1158 & 1612 & 288 & ~1196 & ~1610 & ~1611 & ~1613 & ~1615 & ~514) | (1158 & 1613 & 288 & ~1196 & ~1610 & ~1611 & ~1612 & ~1614 & ~514) | (1158 & 1613 & 288 & ~1196 & ~1610 & ~1611 & ~1612 & ~1615 & ~514) | (1158 & 1610 & ~1196 & ~1366 & ~1611 & ~1612 & ~1613 & ~1614 & ~514) | (1158 & 1610 & ~1196 & ~1366 & ~1611 & ~1612 & ~1613 & ~1615 & ~514) | (1158 & 1611 & ~1196 & ~1366 & ~1610 & ~1612 & ~1613 & ~1614 & ~514) | (1158 & 1611 & ~1196 & ~1366 & ~1610 & ~1612 & ~1613 & ~1615 & ~514) | (1158 & 1612 & ~1196 & ~1366 & ~1610 & ~1611 & ~1613 & ~1614 & ~514) | (1158 & 1612 & ~1196 & ~1366 & ~1610 & ~1611 & ~1613 & ~1615 & ~514) | (1158 & 1613 & ~1196 & ~1366 & ~1610 & ~1611 & ~1612 & ~1614 & ~514) | (1158 & 1613 & ~1196 & ~1366 & ~1610 & ~1611 & ~1612 & ~1615 & ~514) | (1610 & 288 & ~1196 & ~1297 & ~1611 & ~1612 & ~1613 & ~1614 & ~514) | (1610 & 288 & ~1196 & ~1297 & ~1611 & ~1612 & ~1613 & ~1615 & ~514) | (1611 & 288 & ~1196 & ~1297 & ~1610 & ~1612 & ~1613 & ~1614 & ~514) | (1611 & 288 & ~1196 & ~1297 & ~1610 & ~1612 & ~1613 & ~1615 & ~514) | (1612 & 288 & ~1196 & ~1297 & ~1610 & ~1611 & ~1613 & ~1614 & ~514) | (1612 & 288 & ~1196 & ~1297 & ~1610 & ~1611 & ~1613 & ~1615 & ~514) | (1613 & 288 & ~1196 & ~1297 & ~1610 & ~1611 & ~1612 & ~1614 & ~514) | (1613 & 288 & ~1196 & ~1297 & ~1610 & ~1611 & ~1612 & ~1615 & ~514) | (1610 & ~1196 & ~1297 & ~1366 & ~1611 & ~1612 & ~1613 & ~1614 & ~514) | (1610 & ~1196 & ~1297 & ~1366 & ~1611 & ~1612 & ~1613 & ~1615 & ~514) | (1611 & ~1196 & ~1297 & ~1366 & ~1610 & ~1612 & ~1613 & ~1614 & ~514) | (1611 & ~1196 & ~1297 & ~1366 & ~1610 & ~1612 & ~1613 & ~1615 & ~514) | (1612 & ~1196 & ~1297 & ~1366 & ~1610 & ~1611 & ~1613 & ~1614 & ~514) | (1612 & ~1196 & ~1297 & ~1366 & ~1610 & ~1611 & ~1613 & ~1615 & ~514) | (1613 & ~1196 & ~1297 & ~1366 & ~1610 & ~1611 & ~1612 & ~1614 & ~514) | (1613 & ~1196 & ~1297 & ~1366 & ~1610 & ~1611 & ~1612 & ~1615 & ~514)"
        logic_terms = []
        for clause in dnf_rules.split('|'):
            lm = [i.strip() for i in clause.strip("() ").split('&')]
            la = []
            lb = []
            for lit in lm:
                if(lit[0]=='~'):
                    lb.append(int(lit[1:]))
                else:
                    la.append(int(lit))
            lc = [i for i in range(config.total_vocab_size) if (i not in la and i not in lb)]
            logic_terms.append(symbolic.GEQConstant(la, lc, lb, 1, 0, 1))

        self.logic_pred = nn.Sequential(
                    nn.ReLU(),
                    nn.BatchNorm1d(config.total_vocab_size),
                    nn.Linear(config.total_vocab_size, len(logic_terms))
                )        
        self.decoder = symbolic.OrList(terms=logic_terms)

    def forward(self, input_visits, position_ids=None, ehr_labels=None, ehr_masks=None, past=None):
        hidden_states = self.transformer(input_visits, position_ids, past)
        code_logits = self.ehr_head(hidden_states, input_visits)
        sig = nn.Sigmoid()
        code_probs = sig(code_logits)
        for (past_visits, past_codes, past_non_codes, current_codes, current_non_codes, output_code, output_value) in self.rules:
            if past_visits == -1:
                pastVisits = torch.tril(torch.ones(self.n_ctx, self.n_ctx, device=input_visits.device), diagonal=-1)
            else:
                pastVisits = torch.zeros(self.n_ctx, self.n_ctx, device=input_visits.device)
                for v in past_visits:
                    if v >= 0:
                        pastVisits[v+1:,v] = 1
                    else:
                        pastVisits[list(range(-v, self.n_ctx)), [i + v for i in range(-v, self.n_ctx)]]
            
            if past_codes or past_non_codes:
                pastMatrix = torch.zeros(self.total_vocab_size, self.total_vocab_size, device=input_visits.device)
                pastBias = torch.zeros(self.total_vocab_size, device=input_visits.device)
                pastMatrix[past_non_codes, output_code] = -1
                if past_codes:
                    pastMatrix[past_codes, output_code] = 1 / len(past_codes)
                else:
                    pastBias[output_code] = 1
            else:
                pastMatrix = torch.tensor([])
                pastBias = torch.tensor([])
            
            if current_codes or current_non_codes:
                currMatrix = torch.zeros(self.total_vocab_size, self.total_vocab_size, device=input_visits.device)
                currBias = torch.zeros(self.total_vocab_size, device=input_visits.device)
                currMatrix[current_non_codes, output_code] = -1
                if current_codes:
                    currMatrix[current_codes, output_code] = 1 / len(current_codes)
                else:
                    currBias[output_code] = 1
            else:
                currMatrix = torch.tensor([])
                currBias = torch.tensor([])

            if pastMatrix.numel() == 0:
                code_probs = torch.where(((input_visits[:,1:] @ currMatrix) + currBias) == 1, output_value, code_probs)
            else:
                past_visits = pastVisits @ input_visits
                if currMatrix.numel() == 0:
                    code_probs = torch.where(((past_visits[:,1:] @ pastMatrix) + pastBias) == 1, output_value, code_probs)
                else:
                    code_probs = torch.where((((past_visits[:,1:] @ pastMatrix) + pastBias) == 1) & (((input_visits[:,1:] @ currMatrix) + currBias) == 1), output_value, code_probs)            
        if ehr_labels is not None:    
            shift_labels = ehr_labels[..., 1:, :].contiguous()
            if ehr_masks is not None:
                code_probs = code_probs * ehr_masks
                shift_labels = shift_labels * ehr_masks
            static_probs = torch.reshape(code_probs, (code_probs.shape[0]*code_probs.shape[1], code_probs.shape[2]))
            preds = self.logic_pred(static_probs)
            output = self.decoder(static_probs, preds, False)
            class_preds, logic_preds = output
            ll = []
            for j, p in enumerate(class_preds.split(1, dim=1)):
                ll += [
                    F.binary_cross_entropy_with_logits(
                        p.squeeze(1), torch.reshape(shift_labels, (code_probs.shape[0]*code_probs.shape[1], code_probs.shape[2])), reduction="none"
                    ).sum(dim=1)
                ]
            pred_loss = torch.stack(ll, dim=1)
            recon_losses, labels = pred_loss.min(dim=1)

            loss = (logic_preds.exp() * (pred_loss + logic_preds)).sum(dim=1).mean()
            loss += recon_losses.mean()
            loss += F.nll_loss(logic_preds, labels)
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
            
        for (past_visits, past_codes, past_non_codes, current_codes, current_non_codes, output_code, output_value) in self.rules:
            if past_visits == -1:
                pastVisits = torch.tril(torch.ones(self.n_ctx, self.n_ctx, device=input_visits.device), diagonal=-1)
            else:
                pastVisits = torch.zeros(self.n_ctx, self.n_ctx, device=input_visits.device)
                for v in past_visits:
                    if v >= 0:
                        pastVisits[v+1:,v] = 1
                    else:
                        pastVisits[list(range(-v, self.n_ctx)), [i + v for i in range(-v, self.n_ctx)]]
            
            if past_codes or past_non_codes:
                pastMatrix = torch.zeros(self.total_vocab_size, self.total_vocab_size, device=input_visits.device)
                pastBias = torch.zeros(self.total_vocab_size, device=input_visits.device)
                pastMatrix[past_non_codes, output_code] = -1
                if past_codes:
                    pastMatrix[past_codes, output_code] = 1 / len(past_codes)
                else:
                    pastBias[output_code] = 1
            else:
                pastMatrix = torch.tensor([])
                pastBias = torch.tensor([])
            
            if current_codes or current_non_codes:
                currMatrix = torch.zeros(self.total_vocab_size, self.total_vocab_size, device=input_visits.device)
                currBias = torch.zeros(self.total_vocab_size, device=input_visits.device)
                currMatrix[current_non_codes, output_code] = -1
                if current_codes:
                    currMatrix[current_codes, output_code] = 1 / len(current_codes)
                else:
                    currBias[output_code] = 1
            else:
                currMatrix = torch.tensor([])
                currBias = torch.tensor([])
                
            if pastMatrix.numel() == 0:
                input_visits[(((input_visits @ currMatrix) + currBias) == 1)] = output_value
            else:
                past_visits = pastVisits[:input_visits.size(1), :input_visits.size(1)] @ input_visits
                if currMatrix.numel() == 0:
                    input_visits[(((past_visits @ pastMatrix) + pastBias) == 1)] = output_value
                else:
                    input_visits[(((past_visits @ pastMatrix) + pastBias) == 1) & (((input_visits @ currMatrix) + currBias) == 1)] = output_value
        
        return input_visits