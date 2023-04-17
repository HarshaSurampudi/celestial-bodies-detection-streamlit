from fastai.vision.all import *

learn_inf = load_learner('celestial.pkl')


pred, pred_idx, probs = learn_inf.predict('052.jpg')

print(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')

print(probs)