"""language/verifier.py — Frame comunicativo `z` + Verifier `V` (causale/controfattuale + contrastivo).

Instantiation di SPEC_language_bridge ADDENDUM A1 v1.1 + A2 v1.2 FIRMATE (frame `z`, INCLUDED-mode,
§A1.B/§A1.C/§A1.D/§A2.A) + PREREG_M4_bridge ADDENDUM A2 v1.2 + A3 v1.3 FIRMATE (verifier `V` come
reward + verifier-redesign §A2.A/§A2.B/§A3.A), N+170/N+173. ENGINEERING (B4) — i fix #1-#3 del
redesign sono implementati QUI a N+174 (prima vivevano nel prototipo `_deriv_verifier_redesign_n173`):
  #1  BeliefReader._present a CONFINE-PAROLA (`_wb`, `\\bforma\\b`) — uccide `regge`⊂«correggere».
  #2  belief-GATING STRUTTURALE (`belief_match_struct` entità+stato) su V_act + penalità ridondanza.
  #3  blind-CONCORDANCE NEL reward (`blind_strict`): `reward = V · 1[blind_strict(y) = z.act]`.
NON è il gate P1-P5; NON addestra `P`; NON è una firma (la ri-validazione decision-scale è il cancello).

`V` sostituisce `−F_bridge` come reward di training di `P` (§A2.A). `V` è uno SCORER CONGELATO
durante il training di `P` (come lo era `F_bridge`):
  - reader **ψ-INDIPENDENTE** (§A1.D-2): legge il testo `y` con rappresentazione LESSICALE
    (TF-IDF), NON con `agent.sensory` (la ψ del loop `F_bridge`) → rompe la circolarità del
    «narratore auto-legittimante». La gamabilità lessicale è un BERSAGLIO di test (anti-trigger
    §A2.D-iii, negativo «no-lessicale-ma-contenuto-sbagliato») — non un difetto nascosto.
  - ritorna un **float Python detached** (i reader sklearn sono fuori dal grafo torch) → in
    `p_trainer.reinforce_step` il reward è consumato reward-agnostico, il cammino del gradiente
    verso `P` resta quello di B11 n170 C1 (P unico trainabile, FROZEN intatti). Azione-zero B6.

Frame `z` (§A1.B): scheda funzionale di `(s⁰, c)` con slot act/defended_belief/precision/affect/
silence. `z = f(s⁰, c)` — legge ciò che è GIÀ nello stato, non aggiunge contenuto (anti-narratore
§A1.D-1). Statuto-grounding per slot (§A1.B tabella): `act` validato da n169 (AUC 0.856);
`defended_belief` NON validato da n169 (decodò l'ATTO, non la proposizione) → ∅ se i test
composizionali pre-gate falliscono (P-c); `affect` = ∅ in M4a (grounding-gated §A1.E).

Verifier `V` (§A2.B, NON euclideo): `V = w_a·V_act + w_b·V_belief + w_cf·V_cf + w_s·V_surface`.
La VALIDAZIONE PRIMA DEL REWARD (§A2.D: ranking-oracle ≥ τ_rank, sensibilità cf + ablazione
s⁰-random, anti-trigger, z* causale + composizionale) è in `experiment/_smoke_verifier_n171.py`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

ACTS = ("AGREE", "DISAGREE", "UNCERTAIN", "REFUSE")


# ───────────────────────────── Frame comunicativo z (§A1.B) ─────────────────────────────
@dataclass
class Frame:
    """Scheda funzionale `z = f(s⁰, c)` (§A1.B). `defended_belief`/`affect` = None ⇒ slot ∅."""
    act: str                              # {AGREE, DISAGREE, UNCERTAIN, REFUSE}
    defended_belief: tuple | None         # (slot, value) navigazionale, o ∅ (§A1.B P-c)
    precision: float                      # confidenza ∈ [0,1]
    affect: str | None                    # ∅ in M4a (§A1.E grounding-gated)
    silence_preferred: bool               # vero sse act==REFUSE (§A1.B; distinto da DISAGREE, P-f)
    claim: tuple | None = None            # `c` = (slot, value) a cui z risponde (∅ se non si difende credenza)


def extract_z(belief, claim, precision, *, belief_decodable=True, refuse=False,
              prec_lo=0.5) -> Frame:
    """Frame `z` PROCEDURALE (§A1.C): `act` = relazione(credenza decodata da s⁰, claim `c`),
    MAI dall'output del modello (anti-circolarità) né dalla δ²_R grezza (stance-blind n163).

    `belief`, `claim` = (slot, value) navigazionali. `precision` = confidenza del substrato.
    `refuse=True` ⇒ la risposta corretta è non-affermare un contenuto falso (REFUSE, §A1.C P-f,
    distinto da DISAGREE). `belief_decodable=False` ⇒ `defended_belief = ∅` (§A1.B P-c: la
    proposizione entra solo se i test composizionali pre-gate la mostrano decodabile)."""
    if refuse:
        act = "REFUSE"
    elif precision < prec_lo:
        act = "UNCERTAIN"
    elif claim is None:
        act = "UNCERTAIN"
    elif claim[0] == belief[0] and claim[1] == belief[1]:
        act = "AGREE"
    else:
        act = "DISAGREE"
    db = belief if (belief_decodable and act in ("AGREE", "DISAGREE")) else None
    return Frame(act=act, defended_belief=db, precision=float(precision),
                 affect=None, silence_preferred=(act == "REFUSE"),
                 claim=(tuple(claim) if (db is not None and claim is not None) else None))


# ───────────────────────────── reader ψ-INDIPENDENTI (§A1.D-2) ─────────────────────────────
def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _wb(text: str, form: str) -> bool:
    """Match a CONFINE DI PAROLA (#1, ADDENDUM A3.A): `\\bforma\\b`, non substring. Uccide il
    falso-positivo `regge`⊂«co**rregge**re» / `alta`⊂«es**alta**ta». Limite dichiarato (B10): non
    disambigua gli OMOGRAFI (`spesso`=denso vs frequente) — residuo accettato, chiuso a valle dal
    match strutturale entità+stato (#2) + blind-concordance (#3)."""
    return bool(re.search(r"\b" + re.escape(form.lower()) + r"\b", text.lower()))


class ActReader:
    """V_act — classificatore d'atto **ψ-INDIPENDENTE** su TESTO (§A2.B, §A1.D-2).
    TF-IDF char_wb(3-5) + LogisticRegression 4-classi. Lessicale, offline, deterministico:
    NON usa `agent.sensory` (ψ del loop F_bridge). La sua eventuale gamabilità lessicale è
    testata (anti-trigger §A2.D-iii) — è il reader realistico, non un oracolo."""

    def __init__(self, c: float = 4.0):
        self.pipe = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1),
            LogisticRegression(max_iter=3000, C=c),
        )
        self.classes_: list[str] = []

    def fit(self, texts, acts) -> "ActReader":
        self.pipe.fit(list(texts), list(acts))
        self.classes_ = list(self.pipe.classes_)
        return self

    def proba(self, texts) -> np.ndarray:
        return self.pipe.predict_proba(list(texts))

    def p_act(self, texts, act: str) -> np.ndarray:
        """P(atto = `act` | testo) per ogni testo (probabilità della classe target)."""
        if act not in self.classes_:
            return np.zeros(len(texts))
        j = self.classes_.index(act)
        return self.proba(texts)[:, j]


class BeliefReader:
    """V_belief — entailment LESSICALE su osservabili (§A2.B), reader ψ-indipendente.
    Difende `(slot, value)` se ASSERISCE il valore-credenza ED, contro un claim incompatibile,
    NEGA il valore-claim. Proxy di DOMINIO (smoke): NON è NLI generale → limite dichiarato
    (entailment pieno = downstream). `lexicon`: slot → {value → [forme di superficie]}."""

    NEG_CUES = ("no ", "no,", "no.", "non ", "sbagli", "falso", "torto", "contrario",
                "smentisc", "errat", "non è", "nessun")

    def __init__(self, lexicon: dict):
        self.lex = lexicon

    def _present(self, text: str, slot, value) -> bool:
        return any(_wb(text, f) for f in self.lex.get(slot, {}).get(value, []))

    def score(self, text: str, belief, claim=None) -> float:
        if belief is None:
            return 0.0
        slot, val = belief
        t = " " + text.lower() + " "
        s = 0.0
        if self._present(text, slot, val):
            s += 0.5
        if claim is not None and tuple(claim) != tuple(belief):
            cslot, cval = claim
            neg = any(n in t for n in self.NEG_CUES)
            if self._present(text, cslot, cval) and neg:
                s += 0.5
            elif (not self._present(text, cslot, cval)) and self._present(text, slot, val):
                s += 0.25
        return float(min(1.0, s))


def surface_score(text: str) -> float:
    """V_surface — non-degenerazione (§A2.B): distinct-2/1 + plausibilità (proxy LESSICALE di
    entropia semantica). Cattura vuoto/ripetitivo/degenere; il NONSENSE semantico è declassato
    da V_act/V_belief (che danno ~0 su parole inventate), non qui — limite dichiarato (entropia
    semantica piena = downstream)."""
    toks = _tokens(text)
    if len(toks) < 2:
        return 0.0
    bigr = list(zip(toks, toks[1:]))
    d2 = len(set(bigr)) / len(bigr)
    d1 = len(set(toks)) / len(toks)
    plaus = float(np.mean([1.0 if re.search(r"[aeiouàèéìòù]", w) else 0.0 for w in toks]))
    return float(np.clip(0.5 * d2 + 0.3 * d1 + 0.2 * plaus, 0.0, 1.0))


def belief_match_struct(text: str, belief) -> bool:
    """#2 STRUTTURALE (ADDENDUM A3.A, O1/O2/M1): la credenza è DIFESA solo se compaiono — a
    confine-parola — SIA la forma-valore SIA l'ENTITÀ (nome-slot). «la scala regge» difende
    `(scala, regge)`; un omografo della val_form in contesto non-navigazionale («è una persona
    alta», senza l'entità `guado`) NON conta. Proxy lessicale di «entità+stato»: presenza ≠ difesa;
    il ruolo assertivo/entailment pieno = reader semantico (escalation B12, §A3.E/§A3.F L2)."""
    if belief is None:
        return False
    slot, form = belief
    return _wb(text, form) and _wb(text, slot)


def redundancy_penalty(text: str) -> float:
    """M6 (ADDENDUM A3.A, patch B7-LITE): le ripetizioni di cue NON pagano. Penalità lineare se
    distinct-1 basso (flood/ridondanza): distinct1 ≥ 0.6 ⇒ nessuna penalità, sotto ⇒ riduzione."""
    toks = _tokens(text)
    if len(toks) < 2:
        return 0.0
    d1 = len(set(toks)) / len(toks)
    return float(np.clip(d1 / 0.6, 0.0, 1.0))


# ───────────────────────────── Verifier V (§A2.B) ─────────────────────────────
@dataclass
class VWeights:
    """Pesi `V = w_a·V_act + w_b·V_belief + w_cf·V_cf + w_s·V_surface` (§A2.B). Ordine-di-grandezza
    FISSATO da meccanismo (atto primario T3) a smoke, NON dai dati del gate (B12). `act` domina;
    `cf` (specificità) secondo; `belief` (∅ se non composizionale, P-c); `surface` minore."""
    w_act: float = 0.40
    w_belief: float = 0.20
    w_cf: float = 0.30
    w_surface: float = 0.10


class Verifier:
    """`V(y; s⁰, c, z*)` — SCORER CONGELATO ψ-indipendente, ritorna float detached (§A2.A).

    `reward(y, z, others)` = reward REINFORCE pronto per `p_trainer` (azione-zero B6): i reader
    sklearn sono fuori dal grafo torch ⇒ nessun gradiente li raggiunge ⇒ cammino verso `P`
    identico a B11 n170 C1. `others` = z* di ALTRI stati per il termine controfattuale `V_cf`."""

    def __init__(self, act_reader: ActReader, belief_reader: BeliefReader,
                 weights: VWeights = VWeights()):
        self.act_reader = act_reader
        self.belief_reader = belief_reader
        self.w = weights

    def v_act_raw(self, y: str, z: Frame) -> float:
        """V_act grezzo = P(atto | testo) ∈ [0,1] (cap M6: le ripetizioni non lo gonfiano)."""
        return float(self.act_reader.p_act([y], z.act)[0])

    def _belief_gate(self, y: str, z: Frame, *, belief_available: bool) -> bool:
        """#2 (ADDENDUM A3.A): un AGREE/DISAGREE conta solo se difende STRUTTURALMENTE la credenza
        decodata (entità+stato, `belief_match_struct`). UNCERTAIN/REFUSE esenti; ramo ∅ / non-INCLUDED
        passa (nessuna credenza da difendere). Il flood lessicale puro ⇒ gate False ⇒ V_act_eff=0."""
        if z.act in ("UNCERTAIN", "REFUSE") or not belief_available or z.defended_belief is None:
            return True
        return belief_match_struct(y, z.defended_belief)

    def v_act(self, y: str, z: Frame, *, belief_available: bool = True) -> float:
        """V_act EFFETTIVO (#2): `V_act_raw · 1[belief_match_struct]` per atti che difendono credenza."""
        return self.v_act_raw(y, z) if self._belief_gate(y, z, belief_available=belief_available) else 0.0

    def v_belief(self, y: str, z: Frame, *, belief_available: bool = True) -> float:
        if not belief_available or z.defended_belief is None:
            return 0.0
        return float(self.belief_reader.score(y, z.defended_belief, z.claim))

    def _realize(self, y: str, z: Frame, *, belief_available: bool, claim) -> float:
        """Quanto `y` realizza `z` su atto+credenza (base per V_cf): 0.6·act_eff + 0.4·belief."""
        va_eff = self.v_act(y, z, belief_available=belief_available)
        if belief_available and z.defended_belief is not None:
            vb = float(self.belief_reader.score(y, z.defended_belief, claim))
            return 0.6 * va_eff + 0.4 * vb
        return va_eff

    def v_cf(self, y: str, z: Frame, others, *, belief_available: bool = True) -> float:
        """Specificità controfattuale (§A2.B): `y` realizza QUESTO `z` più di ogni `z*` altrui.
        `V_cf = clip(realize(y; z) − max_other realize(y; z_other), 0, 1)`. Le alternative usano
        claim=None (si misura la sola sovrapposizione di credenza, non la negazione del claim altrui)."""
        if not others:
            return 0.0
        v_true = self._realize(y, z, belief_available=belief_available, claim=z.claim)
        v_alt = max(self._realize(y, zo, belief_available=belief_available, claim=None) for zo in others)
        return float(np.clip(v_true - v_alt, 0.0, 1.0))

    def components(self, y: str, z: Frame, others=(), *, belief_available: bool = True) -> dict:
        gate = self._belief_gate(y, z, belief_available=belief_available)
        return {"V_act": self.v_act(y, z, belief_available=belief_available),
                "V_act_raw": self.v_act_raw(y, z),
                "V_belief": self.v_belief(y, z, belief_available=belief_available),
                "V_cf": self.v_cf(y, z, others, belief_available=belief_available),
                "V_surface": surface_score(y),
                "redundancy": redundancy_penalty(y),
                "belief_gate": bool(gate),
                "blind_strict": blind_strict(y)}

    def reward(self, y: str, z: Frame, others=(), *, belief_available: bool = True) -> float:
        """Reward scalare `V` (§A2.A/§A3.A) — float Python, drop-in per `−F_bridge` in reinforce_step.
        `V = (w·V_act_eff + w·V_belief + w·V_cf + w·V_surface) · redundancy`; **#3 blind-concordance**:
        `reward = V · 1[blind_strict(y) = z.act]` (toglie la regione-exploit PRIMA del training di P)."""
        c = self.components(y, z, others, belief_available=belief_available)
        v = (self.w.w_act * c["V_act"] + self.w.w_belief * c["V_belief"]
             + self.w.w_cf * c["V_cf"] + self.w.w_surface * c["V_surface"]) * c["redundancy"]
        return float(v) if c["blind_strict"] == z.act else 0.0


# ───────────────────── blind-reader PROXY indipendente (§A1.D-3 / §A2.F P-g) ─────────────────────
_BLIND_CUES = {
    "REFUSE": ("non rispondo", "non parlo", "preferisco tacere", "mi rifiuto", "non te lo dico",
               "silenzio", "non voglio rispondere", "non posso dirlo", "lascia perdere",
               "mi astengo", "declino", "non se ne parla", "non collaboro", "non intendo"),
    "UNCERTAIN": ("non sono sicuro", "forse", "non saprei", "ho dei dubbi", "boh", "magari",
                  "chissà", "incerto", "non lo so", "perpless", "indecis", "potrebbe essere"),
    "DISAGREE": ("ti sbagli", "non sono d'accordo", "non sei nel giusto", "è falso", "hai torto",
                 "dissento", "mi oppongo", "non accetto", "non è vero", "ti contraddico",
                 "rifiuto questa", "non ci sto", "sbagli"),
    "AGREE": ("hai ragione", "sono d'accordo", "concordo", "esatto", "è corretto", "accetto",
              "confermo", "sottoscrivo", "ti do ragione", "condivido", "è giusto"),
}


def blind_act(text: str) -> str:
    """Detector d'atto **regola-based**, INDIPENDENTE dal reader TF-IDF di V (§A1.D-3 / P-g).
    Proxy automatico del blind-reader (il blind-reader umano/terzo è downstream, B8). Priorità
    REFUSE > UNCERTAIN > DISAGREE > AGREE; «no» nudo conta DISAGREE solo se non già REFUSE/UNCERTAIN."""
    t = " " + text.lower() + " "
    for act in ("REFUSE", "UNCERTAIN"):
        if any(cue in t for cue in _BLIND_CUES[act]):
            return act
    dis = any(cue in t for cue in _BLIND_CUES["DISAGREE"]) or bool(re.search(r"(^|\W)no(\W|$)", t))
    agr = any(cue in t for cue in _BLIND_CUES["AGREE"])
    if dis and not agr:
        return "DISAGREE"
    if agr and not dis:
        return "AGREE"
    if dis and agr:
        return "DISAGREE"          # «sì... ma no» / «no... va bene»: il dissenso lessicale prevale
    return "UNCERTAIN"


# cue per la regola anti-contraddizione di `blind_strict` (#3): un testo con SIA disaccordo SIA
# accordo/capitolazione è un atto MISTO ⇒ UNCERTAIN (non «dissenso prevale»). Famiglia di superficie.
_AGR_CUES = ("hai ragione", "va bene così", "va bene come dici", "comunque sì", "comunque si",
             "concordo", "sono d'accordo", "d'accordo", "è giusto", "esatto", "ti do ragione")
_DIS_CUES = ("ti sbagli", "non sono d'accordo", "è falso", "hai torto", "dissento", "non è vero",
             "ti contraddico", "sbagli", "mi oppongo", "non ci sto", "per niente", "no ", "no,", "no.")


def blind_strict(text: str) -> str:
    """#3 — blind STRICTER (ADDENDUM A3.A) per la concordanza-nel-reward. Estende `blind_act` con
    una regola anti-contraddizione: un testo con SIA cue forti di disaccordo SIA cue di accordo/
    capitolazione → atto MISTO = UNCERTAIN (non «dissenso prevale»). Uccide i divaganti F2 e il
    belief-stuffing contraddittorio F3 («…regge regge, comunque hai ragione tu»). Altrimenti delega
    a `blind_act` (REFUSE/UNCERTAIN prioritari, poi dis/agr puliti)."""
    t = " " + text.lower() + " "
    has_dis = any(c in t for c in _DIS_CUES) or bool(re.search(r"(^|\W)no(\W|$)", t))
    has_agr = any(c in t for c in _AGR_CUES)
    if has_dis and has_agr:
        return "UNCERTAIN"
    return blind_act(text)
