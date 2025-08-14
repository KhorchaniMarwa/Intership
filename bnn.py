import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

# 1. Load your dataset (replace with your path)
df = pd.read_csv('stability_dataset.csv')
X = df[['a', 'e', 'i']].values
y = df['is_stable'].values

# 2. Define prior and posterior for Bayesian Dense layer
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])

def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1)),
    ])

# 3. Build the Bayesian Neural Network
model = tf.keras.Sequential([
    tfp.layers.DenseVariational(
        units=16,
        make_prior_fn=prior_trainable,
        make_posterior_fn=posterior_mean_field,
        kl_weight=1/X.shape[0],
        activation='relu'),
    tfp.layers.DenseVariational(
        units=1,
        make_prior_fn=prior_trainable,
        make_posterior_fn=posterior_mean_field,
        kl_weight=1/X.shape[0]),
    tfp.layers.DistributionLambda(lambda t: tfd.Bernoulli(logits=t))
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=lambda y, p_y: -p_y.log_prob(tf.expand_dims(y, axis=-1)),
    metrics=[]
)

# 4. Train (for demo, use a few epochs—adjust as needed)
model.fit(X, y, epochs=30, batch_size=32, verbose=1)

# 5. Prediction & Uncertainty (MC sampling)
def predict_with_uncertainty(model, X, n_samples=100):
    probs = []
    for _ in range(n_samples):
        pred_dist = model(X)
        prob = pred_dist.mean().numpy().flatten()  # mean probability from Bernoulli
        probs.append(prob)
    probs = np.array(probs)
    return probs.mean(axis=0), probs.std(axis=0)

# 6. Example: Predict on new data (replace with your real TNOs)
real_tnos = pd.read_csv('real_tno_catalog.csv')
X_pred = real_tnos[['a', 'e', 'i']].values
mean_probs, std_probs = predict_with_uncertainty(model, X_pred, n_samples=100)

# 7. Save predictions with uncertainty
real_tnos['stability_prob'] = mean_probs
real_tnos['uncertainty'] = std_probs
real_tnos['predicted_label'] = (mean_probs > 0.5).astype(int)
real_tnos.to_csv('tno_stability_predictions.csv', index=False)
print("Saved tno_stability_predictions.csv")