import os

from model._model import Model
from configs import DATA_DIR
from utils.notifier import _notifier


def main():
    alliances = os.listdir(DATA_DIR)
    book_makers = ['tw', 'oversea']
    type_of_bets = ['diff', 'total']

    for alliance in alliances:
        for book_maker in book_makers:
            for type_of_bet in type_of_bets:
                model = Model(
                    alliance=alliance,
                    book_maker=book_maker,
                    type_of_bet=type_of_bet
                )
                result = model.train()
                if result:
                    model.save()
                    _notifier("{} {} {} model saved".format(alliance, book_maker, type_of_bet))


if __name__ == '__main__':
    main()