from TSB_AD.models.AE import AutoEncoder
from TSB_AD.models.IForest import IForest
from TSB_AD.models.LOF import LOF
from TSB_AD.models.LSTMAD import LSTMAD


def build_offline_model(name: str, feats: int, window_size: int, batch_size: int, seed: int):
    if name == "LSTMAD":
        return LSTMAD(window_size=window_size, feats=feats, batch_size=batch_size, epochs=15)
    if name == "AutoEncoder":
        return AutoEncoder(
            slidingWindow=window_size,
            hidden_neurons=[64, 32],
            batch_size=batch_size,
            epochs=15,
        )
    if name == "IForest":
        return IForest(
            slidingWindow=window_size,
            n_estimators=200,
            contamination=0.05,
            max_features=1.0,
            n_jobs=1,
            random_state=seed,
        )
    if name == "LOF":
        return LOF(slidingWindow=window_size, n_neighbors=20, contamination=0.05)
    raise ValueError(f"Unknown offline model: {name}")
