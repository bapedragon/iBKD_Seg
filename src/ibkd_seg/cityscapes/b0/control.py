"""186-update observations including a resumable, incomplete interval."""
from __future__ import annotations

from ...phase1.controllers import GuidanceController
from ..runtime import restore_controller


class StepController:
    def __init__(self,kind,beta):
        self.controller = GuidanceController(kind=kind,beta=beta,threshold=-.02,window=50,warmup_epochs=0)
        self.intervals,self.steps,self.loss_sum,self.samples = 0,0,0.,0
        self.beta = self.controller.beta_for_epoch(1)

    def observe_step(self,raw_loss,batch):
        self.steps += 1
        self.loss_sum += raw_loss*batch
        self.samples += batch
        if self.steps==186:
            self.intervals += 1
            self.controller.observe(self.intervals,self.loss_sum/self.samples,beta_used=self.beta)
            self.beta = self.controller.beta_for_epoch(self.intervals+1)
            self.steps,self.loss_sum,self.samples = 0,0.,0

    def state_dict(self):
        return {"controller":self.controller.state_dict(),"intervals":self.intervals,"steps":self.steps,
                "loss_sum":self.loss_sum,"samples":self.samples,"beta":self.beta}

    def load_state_dict(self,state):
        restore_controller(self.controller,state["controller"])
        for key in ("intervals","steps","loss_sum","samples","beta"):
            setattr(self,key,state[key])


def boundary_checks():
    records = {}
    for kind in ("lg","alg","ibkd"):
        c = StepController(kind,1.)
        for _ in range(185): c.observe_step(1.,16)
        saved = c.state_dict()
        d = StepController(kind,1.)
        d.load_state_dict(saved)
        assert d.steps==185 and d.controller.losses==[]
        c.observe_step(1.,16);d.observe_step(1.,16)
        assert c.state_dict()==d.state_dict() and c.intervals==1 and c.beta==1.
        for _ in range(186): c.observe_step(1.,16)
        assert c.intervals==2
        expected = 1. if kind=="lg" else 0.
        assert c.beta==expected
        records[kind]={"status":"passed","observation_steps":[186,372],
                       "beta_at_373":c.beta,"synthetic_loss_sequence":True}
    # Verify the exact >= (ALG) versus > (iBKD) boundary independently of step averaging.
    for kind,second,active in (("alg",-.08,False),("ibkd",-.04,True)):
        c=GuidanceController(kind=kind,beta=1.,threshold=-.02,window=50,warmup_epochs=0)
        for epoch,loss in enumerate((0.,second),1):
            beta=c.beta_for_epoch(epoch);c.observe(epoch,loss,beta_used=beta)
        assert c.active==active
    return records
