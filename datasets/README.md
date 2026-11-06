# Dataset directory

Full benchmark files are intentionally excluded from Git. Place downloaded data under this directory or point the experiment runner to an external dataset root.

Two expected layouts are included as examples:

```text
datasets/
├── ETT-small/ETTh1.csv
└── weather/weather.csv
```

The evaluation task manifest also uses ETTh2, ETTm1, ETTm2, Electricity, Exchange, PEMS03, PEMS04, PEMS07, and PEMS08. Keep those full files outside the repository when storage is limited.
