import numpy as np
from reciprocal_match.models.calibration import clipped_logit,fit_platt_calibrator

def test_clipped_logit_finite():
    z=clipped_logit(np.array([0.,.5,1.])); assert np.isfinite(z).all() and z[1]==0

def test_platt_probability_range():
    c=fit_platt_calibrator(np.array([.1,.2,.8,.9]),np.array([0,0,1,1])); p=c.predict(np.array([.1,.9])); assert ((p>=0)&(p<=1)).all()
