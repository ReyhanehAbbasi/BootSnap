import numpy as np

def bootstrapped_sampled(Y_train_122,lb):
    label_12=lb.inverse_transform(Y_train_122)
    up_i=[i for i, x in enumerate(label_12) if x == "UP"]
    id_up=np.random.choice(up_i, 4000, replace=False)
    #
    u_i=[i for i, x in enumerate(label_12) if x == "U"]
    id_u=np.random.choice(u_i, 1000, replace=True)
    ui_i=[i for i, x in enumerate(label_12) if x == "UI"]
    id_ui=np.random.choice(ui_i, 1000, replace=True)
    c_i=[i for i, x in enumerate(label_12) if x == "C"]
    id_c=np.random.choice(c_i, 1000, replace=True)
    d_i=[i for i, x in enumerate(label_12) if x == "D"]
    id_d=np.random.choice(d_i, 1000, replace=True)
    us_i=[i for i, x in enumerate(label_12) if x == "US"]
    id_us=np.random.choice(us_i, 1000, replace=True)
    h_i=[i for i, x in enumerate(label_12) if x == "H"]
    id_h=np.random.choice(h_i, 1000, replace=True)
    s_i=[i for i, x in enumerate(label_12) if x == "S"]
    id_s=np.random.choice(s_i, 1000, replace=True)
    c2_i=[i for i, x in enumerate(label_12) if x == "C2"]
    id_c2=np.random.choice(c2_i, 1000, replace=True)
    f_i=[i for i, x in enumerate(label_12) if x == "F"]
    id_f=np.random.choice(f_i, 1000, replace=False)
    c3_i=[i for i, x in enumerate(label_12) if x == "C3"]
    id_c3=np.random.choice(c3_i, 1000, replace=True)
    fp_i=[i for i, x in enumerate(label_12) if x == "FP"]
    id_fp=np.random.choice(fp_i,4000, replace=False)
    indices_bootstraped=np.concatenate((id_up,id_u,id_ui,id_c,id_d,id_us,id_h,id_s,id_c2,id_f,id_c3,id_fp))
    return indices_bootstraped
