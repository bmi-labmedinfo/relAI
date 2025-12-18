import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from ReliabilityPackage.ReliabilityClasses import AE, ReliabilityDetector, DensityPrincipleDetector
from ReliabilityPackage.ReliabilityPrivateFunctions import _train_one_epoch, _compute_synpts_perf, _val_scores_diff_mse, \
    _contains_only_integers, _extract_values_proportionally
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm.auto import tqdm 


# Functions
def _plot_validation_loss_non_blocking(validation_loss, title="Loss"):
    """Show the loss plot without blocking the rest of the code."""
    fig, ax = plt.subplots()
    ax.plot(validation_loss)
    ax.set_xlabel("epochs")
    ax.set_ylabel("Validation Loss")
    ax.set_title(title)

    # Non-blocking show (terminal-friendly)
    plt.show(block=False)
    # Give the GUI event loop a tiny slice so the window renders
    plt.pause(0.001)
    return fig, ax


def train_autoencoder(
    ae,
    training_set,
    validation_set,
    batchsize,
    epochs=1000,
    optimizer_class=torch.optim.Adam,
    optimizer_params=None,
    loss_function=torch.nn.MSELoss(),
):
    """
    Trains the autoencoder model using the provided training and validation sets,
    using tqdm instead of printing epochs, and showing a non-blocking loss plot.
    """
    if optimizer_params is None:
        optimizer_params = {"lr": 1e-4, "weight_decay": 1e-8}

    optimizer = optimizer_class(ae.parameters(), **optimizer_params)

    training_loader = DataLoader(dataset=training_set, batch_size=batchsize, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=batchsize, shuffle=True)

    validation_loss = []

    pbar = tqdm(range(epochs), desc="Training AE", unit="epoch", leave=True)
    for epoch_number in pbar:
        ae.train(True)
        train_loss = _train_one_epoch(
            epoch_number, training_set, training_loader, optimizer, loss_function, ae
        )

        ae.train(False)
        running_vloss = 0.0
        i = 0
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs = vdata
                voutputs = ae(vinputs.float())
                vloss = loss_function(voutputs, vinputs.float())
                running_vloss += vloss

        avg_vloss = (running_vloss / (i + 1)).item() if (i + 1) > 0 else float("nan")
        validation_loss.append(avg_vloss)

        # update tqdm postfix (shows in terminal)
        try:
            train_loss_val = float(train_loss)
        except Exception:
            train_loss_val = train_loss.item() if hasattr(train_loss, "item") else train_loss

        pbar.set_postfix(train_loss=f"{train_loss_val:.6f}", val_loss=f"{avg_vloss:.6f}")

    _plot_validation_loss_non_blocking(validation_loss, title="Validation Loss")
    return ae


def create_and_train_autoencoder(
    training_set,
    validation_set,
    batchsize,
    layer_sizes=None,
    epochs=1000,
    optimizer_class=torch.optim.Adam,
    optimizer_params=None,
    loss_function=torch.nn.MSELoss(),
):
    """
    Creates and trains an autoencoder model using the provided training and validation sets,
    using tqdm instead of printing epochs, and showing a non-blocking loss plot.
    """
    if layer_sizes is None:
        dim_input = training_set.shape[1]
        layer_sizes = [dim_input, dim_input + 4, dim_input + 8, dim_input + 16, dim_input + 32]

    ae = AE(layer_sizes)

    if optimizer_params is None:
        optimizer_params = {"lr": 1e-4, "weight_decay": 1e-8}

    optimizer = optimizer_class(ae.parameters(), **optimizer_params)

    training_loader = DataLoader(dataset=training_set, batch_size=batchsize, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=batchsize, shuffle=True)

    validation_loss = []

    pbar = tqdm(range(epochs), desc="Training AE", unit="epoch", leave=True)
    for epoch_number in pbar:
        ae.train(True)
        train_loss = _train_one_epoch(
            epoch_number, training_set, training_loader, optimizer, loss_function, ae
        )

        ae.train(False)
        running_vloss = 0.0
        i = 0
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs = vdata
                voutputs = ae(vinputs.float())
                vloss = loss_function(voutputs, vinputs.float())
                running_vloss += vloss

        avg_vloss = (running_vloss / (i + 1)).item() if (i + 1) > 0 else float("nan")
        validation_loss.append(avg_vloss)

        try:
            train_loss_val = float(train_loss)
        except Exception:
            train_loss_val = train_loss.item() if hasattr(train_loss, "item") else train_loss

        pbar.set_postfix(train_loss=f"{train_loss_val:.6f}", val_loss=f"{avg_vloss:.6f}")

    _plot_validation_loss_non_blocking(validation_loss, title="Validation Loss")
    return ae


def compute_dataset_avg_mse(ae, X):
    """
    Compute the average mean squared error (MSE) for a given autoencoder model and dataset.

    :param AE ae: The autoencoder model.
    :param numpy.ndarray X: The dataset of interest

    :return: The average MSE value for the reconstructed samples.
    :rtype: float
    """
    mse = []
    for i in range(len(X)):
        mse.append(mean_squared_error(X[i],  ae((torch.tensor(X[i, :])).float()).detach().numpy()))
    return np.mean(mse)


def generate_synthetic_points(problem_type, predict_func, X_train, y_train, method='GN', k=5):
    """
    Generates synthetic points based on the specified method.

    This function generates synthetic points based on the method specified in "method".
    'GN': the synthetic points are generated from the training set by adding gaussian random noise, with different
    values of variance, to the continous variables, and by randomly extracting, proportionally to their frequencies, the values of binary and integer variables.

    :param str problem_type: The string indicating whether it is a classification or regression problem. Available options: 'classification', 'regression'
    :param callable predict_func: A callable predict function of a classifier.
    :param numpy.ndarray X_train: The training set with shape (n_samples, n_features).
    :param numpy.ndarray y_train: The training target labels.
    :param str method: The method used to generate synthetic points (default: 'GN').
        Currently, only the 'GN' (Gaussian Noise) method is supported.

    :return: A tuple containing two elements:
        - synthetic_data: The synthetic data points generated with the specified method.
        - perf_syn: The associated performance values of the synthetic data points evaluated with the provided predict_func.
    :rtype: tuple
    """
    allowed_methods = ['GN']
    if method not in allowed_methods:
        raise ValueError(f"Invalid value for method. Allowed values are {allowed_methods}.")

    if method == 'GN':
        noisy_data = X_train.copy()
        for j in range(2, 7):
            noisy_data_temp = X_train.copy()
            for i in range(X_train.shape[1]):
                if _contains_only_integers(X_train[:, i]):
                    noisy_data_temp[:, i] = _extract_values_proportionally(X_train[:, i])
                else:
                    noise = np.random.normal(0, j * 0.1, size=X_train.shape[0])
                    noisy_data_temp[:, i] += noise

            noisy_data = np.concatenate((noisy_data, noisy_data_temp))

    perf_syn = _compute_synpts_perf(problem_type, predict_func, noisy_data, X_train, y_train, k)

    return noisy_data, perf_syn


def perc_mse_threshold(ae, validation_set, perc=95):
    """
    Computes the MSE threshold as a percentile of the MSE of the validation set.

    This function computes the MSE threshold as a percentile of the MSE of the validation set
    using an autoencoder model. It calculates the MSE for each sample in the validation set
    and returns the specified percentile threshold.

    :param torch.nn.Module ae: The autoencoder model.
    :param numpy.ndarray validation_set: The validation set with shape (n_samples, n_features).
    :param int perc: The percentile threshold to compute (default: 95).

    :return: The MSE threshold as the specified percentile of the MSE of the validation set.
    :rtype: float
    """
    val_projections = []
    mse_val = []
    for i in range(len(validation_set)):
        val_projections.append(ae((torch.tensor(validation_set[i, :])).float()))
        mse_val.append(mean_squared_error(validation_set[i], val_projections[i].detach().numpy()))

    return np.percentile(mse_val, perc)


def mse_threshold_plot(ae, X_val, y_val, predict_func, metric='f1_score'):
    """
    Generates a plot of performance metrics based on different MSE thresholds (selected as percentiles of the MSE of
    the validation set).

    This function generates a plot of performance metrics based on different Mean Squared Error (MSE) thresholds.
    It computes the number (and percentage) of the reliable and unreliable samples obtained with each threshold, and
    different performance metrics using the `val_scores_diff_mse` function.
    The plot shows the performance metric selected ('metric') (e.g., balanced_accuracy, precision, recall, F1-score,
    MCC, or Brier score) for reliable and unreliable samples at different MSE thresholds, and their number and
    percentage. A slider allows to move the x-axis.

    :param torch.nn.Module ae: The autoencoder used for projection.
    :param array-like X_val: The validation dataset.
    :param array-like y_val: The validation labels.
    :param callable predict_func: The predict function of the classifier.
    :param str metric: The performance metric to display on the plot. Available options: 'balanced_accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'brier_score'. Default is 'f1_score'.

    :return: A Plotly Figure object representing the MSE threshold plot.
    :rtype: go.Figure
    """
    allowed_metrics = ['balanced_accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'brier_score']
    if metric not in allowed_metrics:
        raise ValueError(f"Invalid value for metric. Allowed values are {allowed_metrics}.")

    if metric == 'balanced_accuracy':
        metrx = 0
    elif metric == 'precision':
        metrx = 1
    elif metric == 'recall':
        metrx = 2
    elif metric == 'f1_score':
        metrx = 3
    elif metric == 'mcc':
        metrx = 4
    elif metric == 'brier_score':
        metrx = 5

    mse_threshold_list, rel_scores, unrel_scores, num_unrel, perc_unrel = _val_scores_diff_mse(ae, X_val, y_val,
                                                                                              predict_func)
    perc_rel = ['{:.2f}'.format((1 - perc_unrel[i]) * 100) for i in range(len(perc_unrel))]
    perc_unrel = ['{:.2f}'.format(perc_unrel[i] * 100) for i in range(len(perc_unrel))]
    num_rel = [X_val.shape[0] - i for i in num_unrel]
    percentiles = [i for i in range(2, 100)]
    y_rel = [lst[metrx] for lst in rel_scores]
    y_unrel = [lst[metrx] for lst in unrel_scores]

    htxt_rel = [str('{:.2f}'.format(perf)) for perf in y_rel]
    htxt_unrel = [str('{:.2f}'.format(perf)) for perf in y_unrel]

    # Create figure
    fig = go.Figure()
    fig.update_yaxes(range=[min(y_unrel + y_rel), max(y_unrel + y_rel)])
    # fig.update_xaxes(tickformat=".2e")

    for step in range(len(percentiles)):
        fig.add_trace(
            go.Scatter(
                x=percentiles[:step + 2],
                y=y_rel[:step + 2],
                visible=False,
                name='Reliable ' + metric,
                mode='lines',
                line=dict(color='lightgreen'),
                customdata=[[perc, num] for perc, num in zip(perc_rel[:step + 2], num_rel[:step + 2])],
                hovertemplate='%{y:.3f}<br>Reliable samples: %{customdata[1]} (%{customdata[0]}%)',
            )
        )
    for step in range(len(percentiles)):
        fig.add_trace(
            go.Scatter(
                x=percentiles[:step + 2],
                y=y_unrel[:step + 2],
                visible=False,
                name='Unreliable ' + metric,
                mode='lines',
                line=dict(color='salmon'),
                customdata=[[perc, num] for perc, num in zip(perc_unrel[:step + 2], num_unrel[:step + 2])],
                hovertemplate='%{y:.3f}<br>Unreliable samples: %{customdata[1]} (%{customdata[0]}%)',
            )
        )
    # Create and add slider
    steps = []
    for i in range(int(len(fig.data) / 2)):
        step = dict(
            method="update",
            label=str(percentiles[i]) + "°-P",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": str(metric) + " variation on the validation set at different values of the MSE threshold"},
                  {"x": [percentiles[:i + 2]]},  # Update x-axis data
                  ],
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][int(len(fig.data) / 2) + i] = True
        steps.append(step)

    # Make last traces visible
    fig.data[len(percentiles) - 1].visible = True
    fig.data[2 * len(percentiles) - 1].visible = True
    sliders = [dict(
        active=len(percentiles) - 1,
        # currentvalue={"prefix": "MSE x-limit: "},
        pad={"t": 20},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        hovermode="x unified",
        xaxis=dict(
            # tickformat='.2e',
            title='MSE threshold'
        ),
        title=str(metric) + " variation on the validation set at different values of the MSE threshold"
    )

    return fig


def mse_threshold_barplot(ae, X_val, y_val, predict_func):
    """
    Generates a bar plot of performance metrics based on different MSE thresholds.

    This function generates a bar plot of performance metrics based on different Mean Squared Error (MSE) thresholds
    (selected as percentiles of the MSE of the validation set).
    It computes different scores for the reliable and unreliable samples obtained, and the number and percentage of
    unreliable samples, using the `val_scores_diff_mse` function.
    The bar plot shows the percentage of unreliable samples, as well as various performance metrics (e.g.,
    balanced_accuracy, precision, recall, F1-score,  MCC, or Brier score) for reliable and unreliable samples at each
    MSE threshold.
    A slider allows selecting the MSE threshold and updating the plot accordingly.

    :param torch.nn.Module ae: The autoencoder used for projection.
    :param array-like X_val: The validation dataset.
    :param array-like y_val: The validation labels.
    :param callable predict_func: The predict function of the classifier.

    :return: A Plotly Figure object representing the MSE threshold bar plot.
    :rtype: go.Figure
    """
    mse_threshold_list, rel_scores, unrel_scores, num_unrel, perc_unrel = _val_scores_diff_mse(ae, X_val, y_val,
                                                                                              predict_func)
    percentiles = [i for i in range(2, 100)]

    # Creazione del layout con legenda
    hovertext = ["% Unreliable Validation Set",
                 "Balanced Accuracy Reliable set", "Balanced Accuracy Unreliable set",
                 "Precision Reliable set", "Precision Unreliable set",
                 "Recall Reliable set", "Recall Unreliable set",
                 "F1 Score Reliable set", "F1 Score Unreliable set",
                 "MCC Reliable set", "MCC Unreliable set",
                 "Brier Score Reliable set", "Brier Score Unreliable set", ]

    # Create figure
    fig = go.Figure()
    fig.update_yaxes(range=[0, 1])

    colors = ['black',
              'lightgreen', 'salmon',
              'lightgreen', 'salmon',
              'lightgreen', 'salmon',
              'lightgreen', 'salmon',
              'lightgreen', 'salmon',
              'lightgreen', 'salmon']

    # Add traces, one for each slider step
    for step in range(len(mse_threshold_list)):
        ybar = [perc_unrel[step],
                rel_scores[step][0], unrel_scores[step][0],
                rel_scores[step][1], unrel_scores[step][1],
                rel_scores[step][2], unrel_scores[step][2],
                rel_scores[step][3], unrel_scores[step][3],
                rel_scores[step][4], unrel_scores[step][4],
                rel_scores[step][5], unrel_scores[step][5],
                ]
        format_ybar = ["{:.3f}".format(val) for val in ybar]
        fig.add_trace(
            go.Bar(
                x=['% UR',
                   'R-Bal Accuracy', 'UR-Bal Accuracy',
                   'R-Precision', 'UR-Precision',
                   'R-Recall', 'UR-Recall',
                   'R-f1', 'UR-f1',
                   'R-MCC', 'UR-MCC',
                   'R-brier score', 'UR-brier score'
                   ],
                y=ybar,
                visible=False,
                marker=dict(color=colors),
                name='',
                width=0.8,
                text=format_ybar,
                showlegend=False,
                hovertext=hovertext,
                hoverinfo='text'
            )
        )

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            label=str(i + 2) + "°-P",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "MSE threshold: " + str('{:.4e}'.format(mse_threshold_list[i])) + ": " + str(
                      i + 1) + "°-percentile" +
                            " --- # Unreliable: " + str(num_unrel[i]) +
                            " (" + str('{:.2f}'.format(perc_unrel[i] * 100)) + "%)"}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    # Make 10th trace visible
    fig.data[49].visible = True

    sliders = [dict(
        active=49,
        currentvalue={"prefix": "MSE threshold: "},
        pad={"t": 20},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title="MSE threshold: " + str('{:.4e}'.format(mse_threshold_list[49])) + ": " + str(50) + "°-percentile" +
              " --- # Unreliable: " + str(num_unrel[49]) +
              " (" + str('{:.2f}'.format(perc_unrel[49] * 100)) + "%)"
    )

    return fig


def density_predictor(ae, mse_thresh):
    """
    Creates a DensityPrincipleDetector object for a given autoencoder and MSE threshold.

    This function creates a DensityPrincipleDetector object using the specified autoencoder and MSE threshold.
    The DensityPrincipleDetector is a density-based predictor that assigns reliability scores to samples based on their
    reconstruction error (MSE) compared to the MSE threshold.

    :param AE ae: The autoencoder used for projection.
    :param float mse_thresh: The MSE threshold used for assigning reliability scores.

    :return: A DensityPrincipleDetector object.
    :rtype: DensityPrincipleDetector
    """
    DP = DensityPrincipleDetector(ae, mse_thresh)
    return DP


def create_reliability_detector(problem_type, ae, syn_pts, perf_syn, mse_thresh, perf_thresh, proxy_model='MLP'):
    """
    Creates a ReliabilityDetector object for a given autoencoder, synthetic points, accuracy of the synthetic points,
    MSE threshold, and accuracy threshold.

    This function creates a ReliabilityDetector object using the specified autoencoder, synthetic points, performance values of
    the synthetic points, MSE threshold, and performance threshold. The ReliabilityDetector assigns the density
    reliability of samples based on their reconstruction error (MSE), with respect to the MSE threshold, while it assigns
    the local fit reliability based on the prediction of a model ('proxy_model'), trained on the synthetic points
    labelled as "local-fit" reliable/unreliable according to their associated performance value with respect to the performance
    threshold.

    :param str problem_type: The string indicating whether it is a classification or regression problem. Available options: 'classification', 'regression'
    :param AE ae: The autoencoder used for projection.
    :param array-like syn_pts: The synthetic points used for training the "local-fit" reliability predictor.
    :param array-like perf_syn: The performance scores corresponding to the synthetic points.
    :param float mse_thresh: The MSE threshold used for assigning the density reliability scores.
    :param float perf_thresh: The performance threshold used for assigning the "local-fit" reliability scores.
    :param str proxy_model: The type of proxy model used for training the "local-fit" reliability predictor.
        Available options: 'MLP', 'tree'. Default is 'MLP' (Multi-Layer Perceptron).

    :return: A ReliabilityPackage object.
    :rtype: ReliabilityDetector
    """
    allowed_proxy_model = ['MLP', 'tree']
    if proxy_model not in allowed_proxy_model:
        raise ValueError(f"Invalid value for proxy_model. Allowed values are {allowed_proxy_model}.")
    y_syn_pts = []
    for i in range(len(perf_syn)):
        if problem_type == 'classification':
            y_syn_pts.append(1) if perf_syn[i] >= perf_thresh else y_syn_pts.append(0)
        elif problem_type == 'regression':
            y_syn_pts.append(1) if perf_syn[i] <= perf_thresh else y_syn_pts.append(0)

    if proxy_model == 'MLP':
        clf = MLPClassifier(activation="tanh", random_state=42, max_iter=1000).fit(syn_pts, y_syn_pts)
    elif proxy_model == 'tree':
        clf = tree.DecisionTreeClassifier(random_state=42).fit(syn_pts, y_syn_pts)

    RP = ReliabilityDetector(ae, clf, mse_thresh)

    return RP


def compute_dataset_reliability(RD, X, mode='total'):
    """
    Computes the reliability of the samples in a dataset

    This function computes the density/local-fit/total reliability of the samples in the X dataset, based on the mode
    specified.
    :param ReliabilityDetector RD: A ReliabilityPackage object.
    :param array-like X: the specified dataset
    :param str mode: the type of reliability to compute; Available options: 'density', 'local-fit', 'total'. Default is 'total'
    :return: a numpy 1-D array containing the reliability of each sample (1 for reliable, 0 for unreliable)
    :rtype: numpy.ndarray
    """
    if mode == 'total':
        return np.asarray([RD.compute_total_reliability(x) for x in X])
    elif mode == 'density':
        return np.asarray([RD.compute_density_reliability(x) for x in X])
    elif mode == 'local-fit':
        return np.asarray([RD.compute_localfit_reliability(x) for x in X])
