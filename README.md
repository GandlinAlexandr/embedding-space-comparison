<a name="readme-top"></a>


[license-shield]: https://img.shields.io/github/license/GandlinAlexandr/embedding-space-comparison.svg?style=for-the-badge
[license-url]: https://github.com/GandlinAlexandr/embedding-space-comparison/blob/main/LICENSE

[![MIT][license-shield]][license-url]
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[!['Black'](https://img.shields.io/badge/code_style-black-black?style=for-the-badge)](https://github.com/psf/black)

<h1 align="center">Геометрические методы сравнения пространств эмбеддингов</h1>
  
  <p align="center">
    Проект в рамках ВКР магистерской программы "Искусственный Интеллект"
  </p>

<details>
  <summary>Содержание</summary>
  <ol>
    <li><a href="#описание-проекта">Описание проекта</a></li>
    <li><a href="#технологии">Технологии</a></li>
    <li><a href="#содержание-проекта">Содержание проекта</a></li>
      <ul><li><a href="#примеры-запуска">Примеры запуска</a></li></ul>
    <li><a href="#лицензия">Лицензия</a></li>
    <li><a href="#контакты">Контакты</a></li>
  </ol>
</details>


# Описание проекта

Данный проект предназначен для разработки метрики геометрического сравнения эмбеддингов нейросетей, основанной на ранге матрицы отображения.

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

# Технологии

Для реализации проекта используются следующие технологии:

* [![Colab][Colab]][Colab-url]
* [![Python][Python.org]][Python-url]
  * [![Matplotlib][Matplotlib.org]][Matplotlib-url]
  * [![Numpy][Numpy.org]][Numpy-url]
  * [![Pandas][Рandas.pydata.org]][Pandas-url]
  * [![Pytorch][Pytorch]][Pytorch-url]
  * [![scikit-learn][scikit-learn]][scikit-learn-url]
  * [![Spacy][SpaCy]][Spacy-url]


<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

# Содержание проекта

## Примеры запуска

Все примеры запуска показаны для `Power Shell`.

A) Пример пошагового запуска и оценки основных метрик данной работы представлен ниже.

<details>
  <summary><b>▶ Пример запуска</b></summary>

1. Получить эмбеддинги на трейне:
   ```powershell
    python -m scripts.run_extract_embeddings `
      --dataset cifar10 `
      --data_root .\data `
      --split train `
      --output_dir .\data\embeddings\cifar10_train `
      --models resnet18,resnet50,wide_resnet50_2,vgg11,vgg16,vgg19,vit_b_16,vit_b_32,vit_l_16 `
      --batch_size 128
   ```
   Итог: Для каждой указанной модели сохраняются эмбеддинги объектов обучающей выборки CIFAR-10. На выходе формируется набор файлов с представлениями, который далее используется как обучающая часть для downstream-оценки и, при необходимости, для других экспериментов с эмбеддингами..
2. Получить эмбеддинги на тесте:
   ```powershell
    python -m scripts.run_extract_embeddings `
      --dataset cifar10 `
      --data_root .\data `
      --split test `
      --output_dir .\data\embeddings\cifar10_test `
      --models resnet18,resnet50,wide_resnet50_2,vgg11,vgg16,vgg19,vit_b_16,vit_b_32,vit_l_16 `
      --batch_size 128
   ```
   Итог: Для каждой указанной модели сохраняются эмбеддинги объектов тестовой выборки CIFAR-10. Эти представления далее используются для вычисления парных метрик между моделями и для оценки качества самих эмбеддингов на downstream-задаче.
3. Downstream train -> test:
   ```powershell
    python -m scripts.run_compute_downstream_scores `
      --train_embeddings_dir .\data\embeddings\cifar10_train `
      --test_embeddings_dir .\data\embeddings\cifar10_test `
      --data_root .\data `
      --out_json .\data\downstream\cifar10_linear_probe.json `
      --task_name cifar10_linear_probe
   ```
   Итог: Для эмбеддингов каждой модели обучается простой downstream-классификатор на train-части и оценивается на test-части. На выходе получается JSON-файл с итоговыми downstream-результатами моделей, который служит опорным сигналом качества представлений и затем используется для оценки того, насколько хорошо парные и одиночные метрики согласуются с реальной разницей в качестве моделей.

4. Посчитать метрики на тестовой выборке:

   а. для антисимметричных версий метрик
   ```powershell
    python -m scripts.run_compute_embedding_metrics `
      --embeddings_dir .\data\embeddings\cifar10_test `
      --out_dir .\data\experiments\exp01_antisym_cifar10_09-03-2026\metric_matrices\cifar10_test `
      --include local_map_rank_linear_knn_k10,local_map_rank_linear_knn_k10_antisym,local_map_rank_linear_knn_k5_antisym,local_map_rank_linear_knn_k20_antisym,local_map_rank_linear_knn_k40_antisym,local_map_rank_linear_eps_percentile_5_antisym,local_map_rank_linear_eps_percentile_10_antisym,local_map_rank_linear_eps_percentile_20_antisym,local_map_rank_multiscale_knn_mean_antisym,local_map_rank_rff_knn_k10_antisym `
      --seed 42 `
      --incremental
   ```
   
   b. для семмитричных версий метрик
   ```powershell
    python -m scripts.run_compute_embedding_metrics `
      --embeddings_dir .\data\embeddings\cifar10_test `
      --out_dir .\data\experiments\exp02_sym_cifar10_10-03-2026\metric_matrices\cifar10_test `
      --include local_map_rank_linear_knn_k10,local_map_rank_linear_knn_k10_sym,local_map_rank_linear_knn_k5_sym,local_map_rank_linear_knn_k20_sym,local_map_rank_linear_knn_k40_sym,local_map_rank_linear_eps_percentile_5_sym,local_map_rank_linear_eps_percentile_10_sym,local_map_rank_linear_eps_percentile_20_sym,local_map_rank_multiscale_knn_mean_sym,local_map_rank_rff_knn_k10_sym `
      --seed 42 `
      --incremental
   ```
   Итог: Для каждой выбранной метрики строится матрица попарных сравнений моделей на тестовых эмбеддингах. Иными словами, для каждой пары моделей получается численное значение метрики, отражающее степень их различия или сходства в выбранном протоколе. На выходе формируются готовые матрицы метрик, которые затем можно сопоставлять с реальными различиями в downstream-качестве моделей.
   
   Важно, что попарные метрики посчитаются по **всем** парам эмбеддингов, которые находятся в `embeddings_dir`. Можно добавлять модели, уже посчитанные метрики пересчитываться не будут. Но убрать из матрицы уже посчитанные модели - пока не реализовано. Аналогично для всякого расчёта любых метрик.

5. Запустить оценку метрик посредством результатов работы MLP:

   а. для антисимметричных версий метрик
   ```powershell
    python -m scripts.run_evaluate_metrics `
      --metrics_dir .\data\experiments\exp01_antisym_cifar10_09-03-2026\metric_matrices\cifar10_test `
      --downstream_json .\data\experiments\exp01_antisym_cifar10_09-03-2026\downstream\cifar10_linear_probe.json `
      --out_csv .\data\experiments\exp01_antisym_cifar10_09-03-2026\reports\cifar10_eval_signed.csv `
      --eval_protocol delta_signed `
      --plots_dir .\data\experiments\exp01_antisym_cifar10_09-03-2026\plots\cifar10_test_signed `
      --plots_mode alltasks `
      --plots_ext png
   ```
   
   b. для симметричных версий метрик
   ```powershell
    python -m scripts.run_evaluate_metrics `
      --metrics_dir .\data\experiments\exp02_sym_cifar10_10-03-2026\metric_matrices\cifar10_test `
      --downstream_json .\data\experiments\exp02_sym_cifar10_10-03-2026\downstream\cifar10_linear_probe.json `
      --out_csv .\data\experiments\exp02_sym_cifar10_10-03-2026\reports\cifar10_eval.csv `
      --eval_protocol abs `
      --plots_dir .\data\experiments\exp02_sym_cifar10_10-03-2026\plots\cifar10_test_signed `
      --plots_mode alltasks `
      --plots_ext png
   ```
   Итог: Для каждой парной метрики вычисляется, насколько хорошо её значения согласуются с реальными различиями в downstream-качестве моделей.
На выходе формируется итоговая таблица оценки метрик, в которой для каждой метрики собраны корреляции и вспомогательные показатели качества.
В антисимметричном протоколе проверяется согласованность с направленной разницей качества между моделями, а в симметричном — с абсолютной величиной этой разницы..

6. Получение графиков по результатам оценки метрик:

   а. для антисимметричных версий метрик
   ```powershell
    python -m scripts.plot_metric_summary `
      --eval_csv .\data\experiments\exp01_antisym_cifar10_09-03-2026\reports\cifar10_eval_signed.csv `
      --out_dir .\data\experiments\exp01_antisym_cifar10_09-03-2026 `
      --dataset cifar10 `
      --protocol "Δacc" `
      --out_name metrics_summary_antisym.png
   ```
   
   b. для симметричных версий метрик
   ```powershell
    python -m scripts.plot_metric_summary `
      --eval_csv .\data\experiments\exp02_sym_cifar10_10-03-2026\reports\cifar10_eval.csv `
      --out_dir .\data\experiments\exp02_sym_cifar10_10-03-2026 `
      --dataset cifar10 `
      --protocol "|Δacc|" `
      --out_name metrics_summary_sym.png
   ```
   Итог: Строятся итоговые сравнительные графики, на которых видно, какие парные метрики лучше всего согласуются с downstream-различиями моделей. Эти визуализации позволяют быстро сравнить варианты метрик между собой и выделить наиболее сильные конфигурации по выбранному протоколу оценки.
</details>

B) Ниже представлены примеры команд запуска вывода графиков, связанных непосредственно с матрицей отображения.

<details>
  <summary><b>▶ Пример запуска</b></summary>

  1. Общая агрегация по всем метрикам
   ```powershell
   python -m scripts.run_diagnose_local_map `
      --artifacts_dir .\data\experiments\exp06_antisym_diagn_cifar10_18-03-2026\metric_matrices\cifar10_test `
      --out_dir .\data\experiments\exp06_antisym_diagn_cifar10_18-03-2026\diagnostics\cifar10_test\summary `
      --degenerate_threshold 0.01
   ```
   Итог: Строятся графики, тражающие качество отображения. Четыре главных графика:
   * Стабильность ранга
   * Ошибка решения линейного уравнения
   * Одновременная вырожденность
   * Спектр сингулярных значений
   
   А также дополнительные графики для каждой метрики в отдельности:
   * Гистограмма рангов
   * Распределение ошибок решения
   * Вырожденность по направлениям (по парам)
   * Одновременная вырожденность (по парам)

2. Данне по конкретной паре моделей
   ```powershell
   python -m scripts.run_diagnose_local_map `
      --artifacts_path .\data\experiments\exp06_antisym_diagn_cifar10_18-03-2026\metric_matrices\cifar10_test\local_map_rank_linear_knn_k10_antisym_artifacts.npz `
      --model_a resnet50 `
      --model_b vit_b_16 `
      --out_dir .\data\experiments\exp06_antisym_diagn_cifar10_18-03-2026\diagnostics\cifar10_test\paars\pair_resnet50_vit_b_16 `
      --degenerate_threshold 0.01
   ```
   Итог: графики качества отображения для конкретной пары моделей. Графики следующие:
   * Гистограмма рангов
   * Распределение ошибок решения
   * Boxplot для ошибок решения
   * Сингулярные значения
      * Для отображения $X\to Y$
      * Для отображения $Y\to X$
   Также выводится таблица со значенимями.
</details>

C) Далее прадставлен пример расчёта "одиночных" метрик (метрик качества эмбеддингов) из статьи [Unsupervised Embedding Quality Evaluation](https://proceedings.mlr.press/v221/tsitsulin23a.html).

<details>
<summary><b>▶ Пример запуска</b></summary>

1. Вычисление "одиночных" метрик:
   ```powershell
    python -m scripts.run_compute_single_metrics `
      --embeddings_dir .\data\embeddings\cifar10_test `
      --out_dir .\data\experiments\exp03_single_cifar10_11-03-2026\single_metrics
   ```
   Итог: Для эмбеддингов каждой модели вычисляются одиночные метрики качества представлений, то есть такие характеристики, которые оценивают каждое пространство эмбеддингов само по себе, без попарного сравнения с другими моделями. На выходе получается набор файлов со значениями этих метрик для всех моделей.

2. Запустить оценку метрик качества эмбеддингов посредством результатов работы MLP:

   а. по протоколу `signed`
   ```powershell
    python -m scripts.run_evaluate_single_metrics `
      --single_metrics_dir .\data\experiments\exp03_single_cifar10_11-03-2026\single_metrics `
      --downstream_json .\data\downstream\cifar10_linear_probe.json `
      --out_csv .\data\experiments\exp03_single_cifar10_11-03-2026\reports\single_metric_eval_signed.csv `
      --out_pairs_dir .\data\experiments\exp03_single_cifar10_11-03-2026\reports\single_metric_pairs_signed `
      --protocol signed
   ```
   
   b. по протоколу `abs`
   ```powershell
    python -m scripts.run_evaluate_single_metrics `
      --single_metrics_dir .\data\experiments\exp03_single_cifar10_11-03-2026\single_metrics `
      --downstream_json .\data\downstream\cifar10_linear_probe.json `
      --out_csv .\data\experiments\exp03_single_cifar10_11-03-2026\reports\single_metric_eval_abs.csv `
      --out_pairs_dir .\data\experiments\exp03_single_cifar10_11-03-2026\reports\single_metric_pairs_abs `
      --protocol abs
   ```
   Итог: Значения одиночных метрик переводятся в формат, пригодный для сравнения с downstream-результатами моделей.
   Для этого по значениям одиночных метрик для разных моделей строятся попарные различия, после чего проверяется, насколько эти различия согласуются с разницей в downstream-качестве.
   На выходе получаются:
   * итоговая таблица качества одиночных метрик;
   * вспомогательные попарные таблицы, используемые для дальнейшей визуализации и анализа.

3. Графики для "одиночных" метрик:

   а. по протоколу `signed`
   ```powershell
    python -m scripts.plot_single_metric_summary `
      --single_eval_csv .\data\experiments\exp03_single_cifar10_11-03-2026\reports\single_metric_eval_signed.csv `
      --pairwise_eval_csv .\data\experiments\exp01_antisym_cifar10_09-03-2026\reports\cifar10_eval_signed.csv `
      --single_protocol signed `
      --out_dir .\data\experiments\exp03_single_cifar10_11-03-2026\reports\plots\antisym-pairwise-single_final_comparison `
      --title "CIFAR10 | Δacc" `
      --save_pairwise_table_csv .\data\experiments\exp03_single_cifar10_11-03-2026\reports\antisym-pairwise-single_final_comparison\pairwise_only.csv `
      --save_single_table_csv .\data\experiments\exp03_single_cifar10_11-03-2026\reports\antisym-pairwise-single_final_comparison\single_diff_only.csv `
      --save_best_comparison_csv .\data\experiments\exp03_single_cifar10_11-03-2026\reports\antisym-pairwise-single_final_comparison\best_pairwise_vs_single.csv
   ```
   
   b. по протоколу `abs`
   ```powershell
    python -m scripts.plot_single_metric_summary `
      --single_eval_csv .\data\experiments\exp03_single_cifar10_11-03-2026\reports\single_metric_eval_abs.csv `
      --pairwise_eval_csv .\data\experiments\exp02_sym_cifar10_10-03-2026\reports\cifar10_eval.csv `
      --single_protocol abs `
      --out_dir .\data\experiments\exp03_single_cifar10_11-03-2026\reports\plots\sym-pairwise-single_final_comparison `
      --title "CIFAR10 | |Δacc|" `
      --save_pairwise_table_csv .\data\experiments\exp03_single_cifar10_11-03-2026\reports\sym-pairwise-single_final_comparison\pairwise_only.csv `
      --save_single_table_csv .\data\experiments\exp03_single_cifar10_11-03-2026\reports\sym-pairwise-single_final_comparison\single_diff_only.csv `
      --save_best_comparison_csv .\data\experiments\exp03_single_cifar10_11-03-2026\reports\sym-pairwise-single_final_comparison\best_pairwise_vs_abs.csv
   ```
   Итог: Строятся итоговые визуализации, в которых одиночные метрики качества эмбеддингов напрямую сравниваются с лучшими парными метриками. Это позволяет увидеть, какие подходы лучше объясняют различия в downstream-качестве моделей: парные меры сравнения пространств или одиночные меры качества самих эмбеддингов.
</details>

D) Ниже указаны команды для графиков, сравнивающих одиночные и попарные метрики в оценке качества эмбеддингов.

<details>
<summary><b>▶ Пример запуска</b></summary>

1. Тепловая карта по семействам моделей для "одиночных" метрик:

   а. по протоколу `signed`
   ```powershell
   python -m scripts.plot_pairwise_error_heatmaps `
      --downstream_json .\data\downstream\cifar10_linear_probe.json `
      --single_metrics_dir .\data\experiments\exp03_single_cifar10_11-03-2026\single_metrics `
      --family_map_json .\data\experiments\model_families.json `
      --out_dir .\data\experiments\exp04_heatmap_13-03-2026\reports\family_corr_signed `
      --pairwise_metrics_dir .\data\experiments\exp01_antisym_cifar10_09-03-2026\metric_matrices\cifar10_test `
      --protocol signed `
      --corr_type spearman `
      --title "CIFAR10 signed" `
      --annotate
   ```
   
   b. по протоколу `abs`
   ```powershell
   python -m scripts.plot_pairwise_error_heatmaps `
      --downstream_json .\data\downstream\cifar10_linear_probe.json `
      --single_metrics_dir .\data\experiments\exp03_single_cifar10_11-03-2026\single_metrics `
      --family_map_json .\data\experiments\model_families.json `
      --out_dir .\data\experiments\exp04_heatmap_13-03-2026\reports\family_corr_abs `
      --pairwise_metrics_dir .\data\experiments\exp02_sym_cifar10_10-03-2026\metric_matrices\cifar10_test `
      --protocol abs `
      --corr_type spearman `
      --title "CIFAR10 abs" `
      --annotate
   ```
   
   Файл `model_families.json` разбивает модели на семейства. Пример файла:
   ```json
   {
    	"resnet18": "resnet",
        "resnet50": "resnet",
        "vgg11": "vgg",
        "vgg16": "vgg",
        "vgg19": "vgg",
        "vit_b_16": "vit",
        "vit_b_32": "vit",
        "vit_l_16": "vit",
        "wide_resnet50_2": "resnet"
   }
   ```
   Итог: Строятся графики, показывающие, насколько хорошо одиночные И попарные метрики согласуются с downstream-различиями в разрезе семейств моделей. Такая визуализация позволяет увидеть, на каких архитектурных группах метрика работает лучше или хуже, и помогает анализировать не только общий средний результат, но и устойчивость поведения метрики на разных типах моделей. 
   
   Графики характеристики выборки:
   * Показывает количество пар моделей, сравниваемых между каждой парой семейств моделей.
   * Средняя значение целевой величины ($\Delta\mathrm{accuracy}$ для протокола `signed` и $|\Delta\mathrm{accuracy}|$ для протокола `abc`) по семействам.
   
   Графики по метрикам:
   * Корреляция метрики и целевой величины по семействам моделей - тепловая карта.
   * Абляционные графики изменения корреляции при удалении конкретного семейства моделей - столбчатые диаграммы.

   Для получения графиков только для попарных или только непарных метрик, досточно просто не указывать дирректорию с матрицами для непарных и парных метрик соответственно.
</details>
<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

# Лицензия

Распространяется по лицензии MIT. Дополнительную информацию см. в файле [`LICENSE`][license-url].

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

# Контакты

Гандлин Александр — [Stepik](https://stepik.org/users/79694206/profile)

Ссылка на проект: [https://github.com/GandlinAlexandr/embedding-space-comparison](https://github.com/GandlinAlexandr/embedding-space-comparison)

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

<!-- Раздел ссылок на сайты и миниатюры -->

[Python-url]: https://python.org/
[Python.org]: https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue

[Pandas-url]: https://pandas.pydata.org/
[Рandas.pydata.org]: https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white

[Numpy-url]: https://numpy.org/
[Numpy.org]: https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white

[Colab-url]: https://colab.research.google.com/
[Colab]: https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252

[scikit-learn-url]: https://scikit-learn.org/
[scikit-learn]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white

[Matplotlib-url]: https://matplotlib.org/
[Matplotlib.org]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black

[Spacy-url]: https://spacy.io/
[Spacy]: https://img.shields.io/badge/-spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white

[Pytorch-url]: https://pytorch.org/
[Pytorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
