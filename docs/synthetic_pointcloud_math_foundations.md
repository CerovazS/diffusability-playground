# Fondamenti Matematici del Dataset Sintetico e della Metrica MMD

Questo documento descrive il modello matematico attualmente implementato per la generazione dei dati sintetici in `datamodules/synthetic_pointclouds.py` e per la valutazione metrica in `utils/pointcloud_metrics.py`.

Nota: il documento sotto descrive soprattutto la formulazione legacy point-cloud. Il setting di training predefinito nel repository e' ora point-wise: ogni sample e' un vettore `x in R^D`, mentre le metriche principali usate di default sono SWD e RBF-MMD su sample raw.

## 1. Obiettivo e notazione

Ogni sample del dataset e' una coppia:

- label \(y\in\{0,\dots,C-1\}\),
- point cloud \(X\in\mathbb{R}^{N\times D}\), con \(N\) punti in spazio ambientale \(D\)-dimensionale.

Parametri principali per classe \(y\):

- dimensione intrinseca \(d\),
- dimensione ambientale \(D\),
- numero di modi \(K\),
- separazione `separation`,
- spessore `thickness`,
- distribuzione intrinseca `tail`,
- anisotropia `anisotropy`,
- curvatura `curvature`,
- famiglia geometrica `family` in `{affine_subspace, sine_warp_subspace, mog}`.

## 2. Processo generativo \(p(X,y)\)

Il dataset implementa due regimi:

1. `samples_per_class = null`: lunghezza fissata `num_samples`, classe campionata uniforme per indice.
2. `samples_per_class != null`: dataset bilanciato, con esattamente `samples_per_class` cloud per classe.

### 2.1 Determinismo per indice

Il generatore pseudo-casuale e' ricreato per ogni `idx` con seed deterministico.

- Caso non bilanciato:
  - `seed = base_seed + idx`
  - \(y\sim\text{Uniform}(\{0,\dots,C-1\})\) usando quel seed.
- Caso bilanciato:
  - `class_index = idx // samples_per_class`
  - `sample_index = idx % samples_per_class`
  - `y = class_ids[class_index]`
  - `seed = base_seed + class_index * samples_per_class + sample_index`.

Con questa scelta, \(idx\mapsto(X,y)\) e' stabile tra run e worker.

### 2.2 Campionamento del modo di mixture

Possiamo definire delle generazioni conditional dipendenti da varie classi. Ogni classe ha attributi in comune definiti dalla singola sweep (numero di modi, anisotropia) e alcuni variabili definiti dalla sweep. 
In generale quindi è possibile fare:
- singolo training con diversi classi generativi ad attributi variabili (`classe1` con anisotropia 1.0, `classe 2` con anosotropia 2.0, `classe 3` con anisotropia 3.0) 
- vari training con un'unica classe per ogni training e sweepare sui vari attributi tra i vari training (training 1: un'unica classe generativa con anisotropia 1.0, training 2: un'unica classe generativa con anisotropia 2.0)

Per una classe con \(K\) componenti:

- se `mode_weights = null`: \(k\sim\text{Uniform}(\{0,\dots,K-1\})\),
- altrimenti \(k\sim\text{Categorical}(w_0,\dots,w_{K-1})\).

### 2.3 Variabili base in coordinate intrinseche

Per ogni punto \(i=1,\dots,N\), si campiona \(z_i\in\mathbb{R}^d\) da una distribuzione di coda:

- `gauss`: \(z_i\sim\mathcal{N}(0,I_d)\),
- `laplace`: campionata via inverse-CDF elemento per elemento,
- `student_t`: implementazione tipo \(z = n/\sqrt{\chi^2/\nu}\), con \(\nu\) approssimato all'intero piu' vicino nel codice (nota importante),
- `cauchy_trunc`: \(z=\tan(\pi(u-1/2))\), poi clipping in \([-\text{clip},\text{clip}]\).

### 2.4 Anisotropia intrinseca

Se `anisotropy.enabled=true`, si definisce \(s\in\mathbb{R}^d\) log-spaziato:

\[
s_j \in [\text{min\_scale}, \text{max\_scale}],\quad
s = \text{logspace}(\text{min\_scale},\text{max\_scale},d)
\]

e si applica:

\[
z_i \leftarrow z_i \odot s
\]

(`permute_per_mode` puo' permutare casualmente le coordinate di \(s\) per cambiare l'allineamento senza cambiare lo spettro dei fattori di scala).

### 2.5 Geometria affine: `affine_subspace`

Si campiona una base ortonormale \(U\in\mathbb{R}^{D\times d}\) (QR di matrice gaussiana), \(U^\top U = I_d\).

Il centro del modo \(k\) e':

\[
\mu_k = \left(k-\frac{K-1}{2}\right)\cdot \text{separation}\cdot v,\quad
v=\frac{g}{\|g\|_2},\ g\sim\mathcal{N}(0,I_D)
\]

Ogni punto:

\[
x_i = U z_i + \mu_k + \varepsilon_i,\quad
\varepsilon_i\sim\mathcal{N}(0,\text{thickness}^2 I_D).
\]

Quindi si ottiene una nube su sottospazio affine (dimensione intrinseca \(d\)) con rumore isotropo.

### 2.6 Geometria curva: `sine_warp_subspace`

Come il caso affine, ma prima del mapping lineare si applica warp sinusoidale elemento per elemento:

\[
z_i \leftarrow z_i + \alpha \sin(\omega z_i),
\]

dove \(\alpha=\text{curvature.alpha}\), \(\omega=\text{curvature.freq}\) (se `curvature.enabled=false`, \(\alpha=0\)).

Poi:

\[
x_i = U z_i + \mu_k + \varepsilon_i.
\]

Il termine sinusoidale introduce non-linearita' controllata (curvatura locale).

### 2.7 Famiglia `mog`

Qui non si usa embedding da coordinate intrinseche: si genera direttamente in \(\mathbb{R}^D\).

\[
x_i = \mu_k + \xi_i + \varepsilon_i,\quad
\xi_i\sim\mathcal{N}(0,\sigma_{\text{mog}}^2 I_D),\ 
\varepsilon_i\sim\mathcal{N}(0,\text{thickness}^2 I_D),
\]

con \(\sigma_{\text{mog}}=\text{mog\_diag\_cov}\).

La covarianza effettiva per modo e' isotropa:

\[
(\sigma_{\text{mog}}^2 + \text{thickness}^2) I_D.
\]

## 3. Espansione sweep parametrici (`class_sweeps`)

Una sweep definisce:

- `base`: parametri di partenza,
- `sweep`: mappa `parametro -> lista valori`.

Il dataset espande il prodotto cartesiano delle liste. Se per i parametri \(p_1,\dots,p_r\) le cardinalita' sono \(m_1,\dots,m_r\), le classi generate dalla sweep sono:

\[
\prod_{j=1}^r m_j.
\]

Ogni combinazione diventa una classe distinta con proprio `class_id`.

## 4. Split train/val/test e comparabilita'

Nel `SyntheticPointCloudDataModule`, i tre split usano config uguali salvo seed (default):

- train: `base_seed`,
- val: `base_seed + 1`,
- test: `base_seed + 2`.

Quindi train/val/test sono campioni indipendenti della stessa famiglia parametrica.

## 5. Metriche oggi calcolate

Per ogni classe, in validazione/test, il codice calcola:

- SWD (`split/swd/class_i`),
- Energy distance U-statistic (`split/energy_u/class_i`),
- Feature-MMD (`split/feature_mmd/class_i`),
- media per classe della Feature-MMD (`split/feature_mmd_mean`).

`split` e' `val` o `test`.

### 5.1 SWD (Sliced Wasserstein Distance)

Si appiattiscono tutte le point cloud della classe in un insieme di punti in \(\mathbb{R}^D\), poi:

1. si campionano direzioni unitarie \(\theta_\ell\), \(\ell=1,\dots,L\),
2. si proiettano i punti su 1D: \(u=\langle x,\theta_\ell\rangle\),
3. si calcola la Wasserstein-1 in 1D tra le due distribuzioni proiettate,
4. si media sulle proiezioni.

Formalmente:

\[
\mathrm{SWD}(P,Q)\approx \frac{1}{L}\sum_{\ell=1}^L
W_1\big((\pi_{\theta_\ell})_\# P,\ (\pi_{\theta_\ell})_\# Q\big).
\]

### 5.2 Energy distance (U-statistic)

La metrica base tra cloud nel codice e' Chamfer simmetrica \(d(\cdot,\cdot)\).
Con campioni \(\{X_i\}_{i=1}^n\) (reali) e \(\{Y_j\}_{j=1}^m\) (generati), stimatore unbiased:

\[
\widehat{\mathcal{E}}_u =
\frac{2}{nm}\sum_{i,j} d(X_i,Y_j)
-\frac{1}{n(n-1)}\sum_{i\neq i'} d(X_i,X_{i'})
-\frac{1}{m(m-1)}\sum_{j\neq j'} d(Y_j,Y_{j'}).
\]

Valori piccoli indicano maggiore somiglianza distribuzionale.

## 6. MMD: cosa stiamo calcolando, perche', come

### 6.1 Idea

Vogliamo confrontare distribuzione reale e generata in modo:

- sensibile a differenze geometriche globali,
- meno costoso di confrontare direttamente insiemi di punti con accoppiamenti completi,
- stabile per monitoraggio durante training.

Per questo usiamo una MMD con kernel RBF su feature per-cloud.

### 6.2 Feature per-cloud \(\phi(X)\)

Da ogni cloud \(X=\{x_i\}_{i=1}^N\subset\mathbb{R}^D\) estraiamo un vettore \(f=\phi(X)\):

1. **centroide**: \(\mu=\frac{1}{N}\sum_i x_i\),
2. **spettro log-covarianza**: autovalori \(\lambda_1\ge\dots\ge\lambda_D\) di
   \[
   \Sigma=\frac{1}{N-1}\sum_i (x_i-\mu)(x_i-\mu)^\top,
   \]
   feature \( \log(\lambda_j+\varepsilon)\),
3. **participation ratio**:
   \[
   \mathrm{PR}=\frac{(\sum_j \lambda_j)^2}{\sum_j \lambda_j^2+\varepsilon},
   \]
4. **thickness spettrale** (coda autovalori):
   \[
   \tau=\sqrt{\frac{1}{k}\sum_{j=D-k+1}^{D}\lambda_j+\varepsilon},
   \]
   con \(k\) determinato da `thickness_tail_fraction`,
5. **kurtosi radiale excess**:
   \[
   r_i=\|x_i-\mu\|_2,\quad z_i=\frac{r_i-\bar r}{\sigma_r+\varepsilon},\quad
   \kappa=\frac{1}{N}\sum_i z_i^4-3,
   \]
6. **curvatura locale media** (surface variation):
   per ogni punto, dai \(k\)-NN si ottiene covarianza locale con autovalori \(\tilde\lambda_1\le\dots\le\tilde\lambda_D\), e
   \[
   c_i=\frac{\tilde\lambda_1}{\sum_j \tilde\lambda_j+\varepsilon},\quad
   c=\frac{1}{N}\sum_i c_i.
   \]

Il vettore finale e' concatenazione dei blocchi abilitati.

Nota: e' invariance alla permutazione dei punti; con `include_centroid=true` non e' invariante a traslazioni globali (scelta intenzionale se la posizione dei modi e' informativa).

### 6.3 Standardizzazione feature

Opzionalmente (`standardize_features=true`) si normalizza usando media e std calcolate sul merge real+gen:

\[
\hat f = \frac{f-\mu_f}{\sigma_f}.
\]

Serve a bilanciare scale eterogenee tra blocchi di feature.

### 6.4 Kernel e stima MMD

Con feature \(u,v\in\mathbb{R}^F\), kernel RBF:

\[
k_\gamma(u,v)=\exp(-\gamma\|u-v\|_2^2).
\]

Se `gamma=null`, il codice usa una median heuristic sui quadrati delle distanze cross-set positive:

\[
\gamma = \frac{1}{\mathrm{median}\{ \|u_i-v_j\|_2^2 > 0\} + \varepsilon},
\]

poi scala con `gamma_scale`.

La quantita' stimata e' \(\mathrm{MMD}^2\):

- **unbiased** (default):
  \[
  \widehat{\mathrm{MMD}}^2_u =
  \frac{1}{n(n-1)}\sum_{i\neq j}k(x_i,x_j)+
  \frac{1}{m(m-1)}\sum_{i\neq j}k(y_i,y_j)-
  \frac{2}{nm}\sum_{i,j}k(x_i,y_j),
  \]
- **biased**:
  \[
  \widehat{\mathrm{MMD}}^2_b =
  \frac{1}{n^2}\sum_{i,j}k(x_i,x_j)+
  \frac{1}{m^2}\sum_{i,j}k(y_i,y_j)-
  \frac{2}{nm}\sum_{i,j}k(x_i,y_j).
  \]

Nel codice la funzione ritorna direttamente questo valore (nomenclatura `mmd`).

La metrica aggregata monitorata a livello split e' media aritmetica sulle classi:

\[
\texttt{split/feature\_mmd\_mean}
= \frac{1}{C}\sum_{c=1}^{C}\texttt{split/feature\_mmd/class\_c}.
\]

### 6.5 Proprieta' teoriche rilevanti

1. **IPM in RKHS**: MMD misura differenza tra embedding medi in spazio di Hilbert.
2. **Popolazione non-negativa**: \(\mathrm{MMD}^2\ge 0\).
3. **Caratteristicita' kernel**: con kernel caratteristico (RBF) su feature fissate, \(\mathrm{MMD}=0\) implica uguaglianza di distribuzione nello spazio delle feature.
4. **Stimatore unbiased puo' essere negativo** a campione finito (varianza), pur stimando una quantita' non-negativa.
5. **Sensibilita' a \(\gamma\)**: confronti tra run sono affidabili solo con stesso protocollo (feature, standardizzazione, gamma/gamma_scale, max_clouds, seed, estimator).

## 7. Cosa significa la MMD qui (interpretazione pratica)

Nel progetto corrente la MMD non confronta direttamente la misura su tutti i punti raw, ma la distribuzione dei descrittori geometrici \(\phi(X)\) delle cloud.

In pratica stiamo chiedendo:

"Le cloud generate hanno la stessa distribuzione di statistiche geometriche delle cloud reali?"

Questo e' adatto alla fase di studio della diffusability, perche':

- separa segnali geometrici interpretabili (spettro, curvatura, thickness, code),
- riduce costo computazionale rispetto a distanze cloud-to-cloud complete,
- permette tracking stabile classe-per-classe.

## 8. Limitazioni e caveat attuali

1. **Feature-space dependence**: se \(\phi\) omette aspetti geometrici, la MMD non li vede.
2. **Comparabilita' cross-run**: va mantenuto identico il protocollo metrica.
3. **Student-t**: implementazione attuale usa approssimazione con \(\nu\) arrotondato a intero.
4. **Centroid block**: con `include_centroid=true` il test e' sensibile a traslazioni globali.
5. **Subsampling** (`max_clouds`) introduce varianza aggiuntiva: utile fissare seed per confronti rigorosi.

## 9. Riferimenti al codice

- Generazione dataset: `datamodules/synthetic_pointclouds.py`
- Metriche point-cloud: `utils/pointcloud_metrics.py`
- Orchestrazione metriche val/test: `SiT/eval_runner.py`
