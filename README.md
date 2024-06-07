# MTRFoodDelivery - Wine Quality Python Project

## Visió General del Projecte

MTRFoodDelivery vol ampliar la gamma de serveis i entrar en el profitós negoci vinícola. Com a experts en Intel·ligència Artificial obriran una nova branca d’assistència a enòlegs, sommeliers i vinicultors.

## Instal·lació

1. Clona el repositori.
2. Assegura't de tenir instal·lat Python 3.7+.
3. Instal·la els paquets necessaris si encara no estan disponibles.

### Creació d'un Entorn Virtual

Per assegurar-te que les dependències del projecte no interfereixin amb altres projectes de Python al teu sistema, es recomana utilitzar un entorn virtual. Segueix aquests passos:

1. **Crea un entorn virtual:**

    ```bash
    python -m venv venv
    ```

2. **Activa l'entorn virtual:**

     - A Windows:

         ```bash
         venv\Scripts\activate
         ```

     - A macOS i Linux:

         ```bash
         source venv/bin/activate
         ```

     Un cop activat, l'indicador de la línia de comandes canviarà per indicar que ara estàs treballant dins de l'entorn virtual.

3. **Instal·la els paquets necessaris:**

     Assegura't d'estar a l'arrel del projecte on està ubicat el fitxer `requirements.txt` amb els paquets necessaris llistats. A continuació, executa:

     ```bash
     pip install -r requirements.txt

## Ús

El script principal `main.py` es pot executar amb diversos arguments per personalitzar la simulació:

```bash
python main.py --operation {testModel,qualityAnalysis} --redWinePath <REDWINEPATH> --whiteWinePath <WHITEWINEPATH> --testSize <TESTSIZE>
```

### Arguments

- `--operation`: L'operació a realitzar. Les opcions disponibles són: (obligatori)
    - `testModel`: Prova el model de classificació de vins amb les dades proporcionades.
    - `qualityAnalysis`: Analitza la qualitat dels vins vermells i blancs de les dades proporcionades.
- `--redWinePath`: La ruta al fitxer CSV que conté les dades del vi negre (valor per defecte: `data/winequality/winequality-red.csv`).
- `--whiteWinePath`: La ruta al fitxer CSV que conté les dades del vi blanc (valor per defecte: `data/winequality/winequality-white.csv`).
- `--testSize`: La mida de la mostra per a la prova del model (valor per defecte: `0.3`).

## Exemple

Aquí teniu un exemple d'execució de la prova del model:

```bash
python main.py --operation testModel
```

En executar-ho amb aquesta comanda el programa:
 - Carregarà les dades dels vins vermells i blancs
 - Les preprocessarà
 - Entrenarà un model de classificació
 - Provarà el model amb les dades de prova.

## Autors

Aquest projecte va ser desenvolupat per Pol Rubio Borrego i Lea Cornelis Martinez com a part d'una pràctica de grup per al curs "TÈCNIQUES D’INTEL·LIGÈNCIA ARTIFICIAL" al Tecnocampus Mataró. Les contribucions són benvingudes per millorar l'eficiència i la funcionalitat de la simulació.

---

Per a més detalls sobre la implementació del projecte i l'enunciat de la pràctica, consulteu el document [PDF proporcionat](practica3_2024.pdf).
