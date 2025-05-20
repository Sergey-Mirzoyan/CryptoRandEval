# CryptoRandEval
Программный комплекс для определения криптостойкости ГПСЧ

** Архитектура ПО (планируемая)

CryptoRandEval/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── config/
│   └── settings.yaml
├── input/
│   └── sample_sequence.bin
├── output/
│   └── reports/
├── modules/
│   ├── stats/
│   │   ├── chi2.py
│   │   ├── nist_sts_wrapper.py
│   │   └── entropy.py
│   ├── predictability/
│   │   ├── lstm_predictor.py
│   │   └── gan_analyzer.py
│   ├── state_recovery/
│   │   ├── lcg_reconstructor.py
│   │   └── mt_state_extractor.py
│   ├── pattern_analysis/
│   │   ├── fft_detector.py
│   │   └── matrix_checker.py
│   ├── entropy_source/
│   │   └── entropy_estimator.py
│   └── report_generator/
│       └── pdf_report_generator.py
├── tests/
│   ├── test_chi2.py
│   └── test_lstm_predictor.py
├── docs/
│   ├── architecture_diagram.md
│   └── user_guide.md
└── examples/
    └── run_analysis.py
