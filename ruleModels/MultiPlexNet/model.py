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
        #for outpatient:
        #dnf_rules = "(1817 & ~1169 & ~1818 & ~1819 & ~1820 & ~1821 & ~1822) | (1817 & ~1169 & ~1818 & ~1819 & ~1820 & ~1821 & ~1823) | (1818 & ~1169 & ~1817 & ~1819 & ~1820 & ~1821 & ~1822) | (1818 & ~1169 & ~1817 & ~1819 & ~1820 & ~1821 & ~1823) | (1819 & ~1169 & ~1817 & ~1818 & ~1820 & ~1821 & ~1822) | (1819 & ~1169 & ~1817 & ~1818 & ~1820 & ~1821 & ~1823) | (1820 & ~1169 & ~1817 & ~1818 & ~1819 & ~1821 & ~1822) | (1820 & ~1169 & ~1817 & ~1818 & ~1819 & ~1821 & ~1823) | (1821 & ~1169 & ~1817 & ~1818 & ~1819 & ~1820 & ~1822) | (1821 & ~1169 & ~1817 & ~1818 & ~1819 & ~1820 & ~1823) | (1596 & 1817 & 1823 & 512 & ~1818 & ~1819 & ~1820 & ~1821 & ~1822) | (1596 & 1818 & 1823 & 512 & ~1817 & ~1819 & ~1820 & ~1821 & ~1822) | (1596 & 1819 & 1823 & 512 & ~1817 & ~1818 & ~1820 & ~1821 & ~1822) | (1596 & 1820 & 1823 & 512 & ~1817 & ~1818 & ~1819 & ~1821 & ~1822) | (1596 & 1821 & 1823 & 512 & ~1817 & ~1818 & ~1819 & ~1820 & ~1822)"

        #for inpatient:
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
        if ehr_labels is not None:    
            shift_labels = ehr_labels[..., 1:, :].contiguous()
            if ehr_masks is not None:
                code_logits = code_logits * ehr_masks
                shift_labels = shift_labels * ehr_masks
            static_probs = torch.reshape(code_logits, (code_logits.shape[0]*code_logits.shape[1], code_logits.shape[2]))
            preds = self.logic_pred(static_probs)
            output = self.decoder(static_probs, preds, False)
            class_preds, logic_preds = output
            ll = []
            for p in class_preds.split(1, dim=1):
                ll += [
                    F.binary_cross_entropy_with_logits(
                        p.squeeze(1), torch.reshape(shift_labels, (code_logits.shape[0]*code_logits.shape[1], code_logits.shape[2])), reduction="none"
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
        preds = self.logic_pred(input_visits[:, -1, :])
        input_visits[:, -1, :] = self.decoder.sample(input_visits[:, -1, :], preds)
        return input_visits