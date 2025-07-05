class BaseStrategy:
    def generate_signals(self, data):
        """
        Prend un DataFrame de prix et retourne un DataFrame de signaux (ex: 1 = achat, -1 = vente, 0 = rien).
        """
        raise NotImplementedError("La méthode generate_signals() doit être implémentée par la sous-classe.")
        