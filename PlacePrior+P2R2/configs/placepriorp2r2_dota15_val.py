# ---------------------------------------------------------------
# DOTA-v1.5 evaluation config
# Same model and weights as placepriorp2r2_dota15.py, but the test
# loader/evaluator are switched to the trainval split for computing mAP.
_base_ = ['./placepriorp2r2_dota15.py']

test_dataloader = _base_.val_dataloader
test_evaluator = dict(type='DOTAMetric', metric='mAP', _delete_=True)
