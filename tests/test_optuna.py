
def test_run():
    import optuna

    def objective(trial):
        x = trial.suggest_uniform('x', -3.0, 3.0)
        return x ** 2

    n_trials = 10
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)
    assert len(study.trials) == n_trials
