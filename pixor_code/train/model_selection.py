from torch import load
from structures.addition_net_structures import Period


# ## Choose th best model
def get_best_model(mt, dataloader):
    """
    :param mt: PixorModel() object
    :param dataloader: Dict with field val
    :return:
    """
    period = Period.validate

    for pth in mt.models_path.glob('t*.pth'):
        mt.load_model(mt.models_path / pth)
        mt.train_model(dataloader, Period.validate)

    history = mt.summary[str(period)].history
    print(f'{period:>8} :: sum: {history.sum_loss[-1]:.5f} | cls: {history.cls_loss[-1]:.5f} | reg: {history.reg_loss[-1]:.5f}')

