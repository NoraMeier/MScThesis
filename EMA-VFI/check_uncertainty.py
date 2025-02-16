from Trainer import Model

model = Model(-1, dropout_flowest=True)
model.load_model('drop_flowest/ours_small_0', custom=True)
model.eval()
