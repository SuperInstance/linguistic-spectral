# linguistic-spectral

**Linguistic spectral analysis — genre detection, author attribution, anomaly detection, and language identification via graph Laplacian conservation.**

Language has a transition graph: bigram and word transition probabilities define edge weights. Build a tension graph Laplacian for text, measure conservation, and use spectral properties for classification. Different genres, authors, and languages produce distinct spectral signatures.

## What This Gives You

- **Text → transition graph** — bigram (character) or word-level transition matrices
- **Genre detection** — poetry, technical, fiction, news, dialogue classification
- **Author attribution** — distinguish authors by spectral fingerprint
- **Anomaly detection** — random words, code injection, language switches
- **Language detection** — English, Spanish, French, German, Chinese
- **Spectral fingerprinting** — conservation ratio, spectral gap, effective dimension

## Quick Start

```bash
pip install numpy matplotlib
python linguistic_spectral.py
```

## How It Fits

Part of the SuperInstance ecosystem:

- **[code-conservation](https://github.com/SuperInstance/code-conservation)** — Source code spectral analysis
- **linguistic-spectral** — Natural language spectral analysis (this repo)

## License

MIT
