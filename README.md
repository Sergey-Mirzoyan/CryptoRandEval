# CryptoRandEval
Программный комплекс для определения криптостойкости ГПСЧ

**Первичная схема архитектуры ПО**

CryptoRandEval/
├── input/

│       └── sequence.bin         # входная последовательность

├── modules/

│       ├── stats/

│       │       ├── chi2.py

│       │       ├── nist_sts_wrapper.py

│       │       └── entropy.py

│       ├── predictability/

│       │       ├── base.py                  # общий интерфейс (Predictor)

│       │       ├── neural_predictor.py      # Pasted_Text_1751980379468.txt

│       │       ├── block_regressor.py       # КС_блоками.txt

│       │       └── statistical_analyzer.py  # общие функции хи-квадрат

│       ├── state_recovery/

│       │       ├── lcg_reconstructor.py

│       │       └── mt_state_extractor.py

│       ├── pattern_analysis/

│       │       ├── fft_detector.py

│       │       └── matrix_checker.py

│       ├── entropy_source/

│       │       └── entropy_estimator.py

│       └── report_generator/

│       └── pdf_report_generator.py

└── output/

        ── crypto_rand_report.pdf