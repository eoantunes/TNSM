from PIL import Image
import numpy as np

###   Para gerar uma nova matriz basta editar os valores I e J abaixo
###   Dimensão dos cenários:   12,15 - 20,25 - 24,30 - 36,45 - 40,50 - 60,75 - 72,90 - 108,135 - 120,150 - 216,270 - 360,450 - 540,675
I,J = 108,135

image = Image.open("brumadinho_1350x1080pb.png")
imagePB = image.convert(mode="L")
imagePB = np.asarray(imagePB)
sh = imagePB.shape
II = sh[0]
JJ = sh[1]

img = np.zeros((II, JJ))
for i in range(II):
    for j in range(JJ):
        if imagePB[i][j] == 255:
            img[i][j] = 1

px = II/I

#####   Matriz de posições aptas a instalação de eNodeBs: Mij   #####
Mij = np.zeros((I, J))
for i in range(II):
    for j in range(JJ):
        if img[i][j] == 1 and Mij[int(i//px)][int(j//px)] == 0:
            Mij[int(i//px)][int(j//px)] += 1

#####   Salva em ARQUIVO para outras análises   #####
Mij = np.asarray(Mij)
np.savetxt('matriz/M{}{}.txt'.format(I,J), Mij, fmt="%d")