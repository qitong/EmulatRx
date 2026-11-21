-- synthetic exclude criterion: prior_systemic_therapy == 1
SELECT stay_id FROM synthetic.cohort WHERE prior_systemic_therapy == 1;
