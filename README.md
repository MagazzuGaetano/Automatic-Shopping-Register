# Automatic-Shopping-Register
Un ipotetico sistema per la text-detection di scontrini

![Blank diagram (1)](https://user-images.githubusercontent.com/30373288/205464548-a733233b-7498-4985-bd76-1ad74168d555.png)

# Testo riconosciuto con Pytesseract:
rASTICCERIA GELATERIA PRESTICE

RUTOMATIC SERVICE S. R. L,

V. LE SAN MARTINO. 168 MESSINA

ARRIVEDERCI E GRAZIE

DOCUMENTO COMMERCIALE
DI VENDITA O PRESTAZIONE

DESCRIZIONE IVA
2 X 2, 20

CONO GELATO 2,2 10%
TOTALE COMPLESSIVO

DI CUI IVA

PAGAMENTO CONTANTE
PAGAMENTO ELETTRONICO
NON RISCOSSO

RESTO
IMPORTO PAGATO

RT 3CIPF101716

CNONTE

EURO

4: 40

40

0. 40

4., 40
0. 00
D. D0
O. a0
4. 40

440


# LIMITAZIONI E PROBLEMI DI SEGMENTAZIONE:
 1) lo sfondo deve essere senza altri oggetti vicini allo scontrino o più grandi di esso
 2) inoltre lo sfondo deve essere più scuro dello scontrino
 3) scontrini deformati accartocciati e con rotazioni estreme sono problematici
 4) casi di illuminazione estremi non sono gestiti

# PROBLEMI DI RICONOSCIMENTO DEI CARATTERI:
 i caratteri sono troppo rovinati e distorti causando problemi nel riconoscimento del testo
