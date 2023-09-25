import pandas as pd
import matplotlib.pyplot as plt

avaliacao = pd.read_csv("arquivos\\files\\report.csv")
print(avaliacao)

resultado = avaliacao.groupby(["Frame"]).sum()
print(resultado)

resultado.plot.bar()
plt.show()