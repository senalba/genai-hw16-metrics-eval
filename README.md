# README — Оцінювання GenAI (текст)

## Мета
Оцінити якість коротких відповідей, згенерованих OpenAI, за стандартними метриками для **тексту** (BLEU, ROUGE). Референсні відповіді були отримані на основі статей з вікіпедії

---

## Структура проєкту
```
.
├─ evaluate_and_ask.py           # запит до OpenAI + BLEU/ROUGE + звіт JSON
├─ environment.yml               # середовище conda
├─ .env                          # містить OPENAI_API_KEY
├─ data/
│  ├─ questions.csv              # запитання (джерело)
│  └─ my_short_answers.csv       # еталонні короткі відповіді
└─ out/
   ├─ openai_answers.json
   └─ report.metrics.json
```

---

## Вимоги та встановлення
1) Створити та активувати середовище:
```bash
mamba/conda env create -f environment.yml
conda activate genai-hw16-eval
```

2) Файл `.env` у корені:
```
OPENAI_API_KEY=sk-...
```

---

## Запуск
```bash
python evaluate_and_ask.py   --questions data/questions.csv   --gold data/my_short_answers.csv   --outdir out   --model gpt-4.1-mini   --max-tokens 80   --temperature 0.0
```

**Вихідні файли:**
- `out/openai_answers.json` — пари `{question, openai_answer}`  
- `out/report.metrics.json` — один запис на приклад із полями:  
  `question, reference, prediction, bleu, rouge1_f1, rouge1_p, rouge1_r, rouge2_f1, rouge2_p, rouge2_r, rougeLsum_f1, rougeLsum_p, rougeLsum_r`

---

## Метрики (текст)
- **BLEU** (sacrebleu): чутливий до точних збігів n-грам.
- **ROUGE-1/2/Lsum**: покриття уні/біграм та узгодженість послідовностей (стійкіше до перефразування).

## Повна таблиця (усі відповіді та метрики)

| question                                        | reference                                                                                                                                | openai_answer                                                                                                                                                                                                                                                         |     bleu |   rouge1_f1 |   rouge2_f1 |   rougeLsum_f1 |
|:------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------:|------------:|------------:|---------------:|
| What is the path integral formulation in QFT?   | Computes amplitudes by summing over all field configurations weighted by exp(iS/ħ); equivalent to the operator formalism.                | The path integral formulation in quantum field theory expresses quantum amplitudes as sums over all possible field configurations, weighted by the exponential of the action, providing a powerful method to compute correlation functions and transition amplitudes. |  3.71253 |    0.415094 |   0.196078  |       0.339623 |
| What is the renormalization group in QFT?       | Describes how couplings run with energy scale via beta functions by systematically integrating out short‑distance modes.                 | The renormalization group in quantum field theory studies how physical parameters like coupling constants change with energy scale, helping to understand behavior of interactions at different distances or energies.                                                |  5.07262 |    0.26087  |   0.0909091 |       0.26087  |
| What is gauge symmetry in QFT?                  | A local redundancy of the field description; enforcing it introduces gauge fields and dictates the interaction structure.                | Gauge symmetry in quantum field theory is a type of symmetry where certain field transformations leave the physical predictions unchanged, reflecting redundant descriptions and leading to conserved charges and fundamental forces.                                 |  1.67943 |    0.375    |   0         |       0.208333 |
| What is spontaneous symmetry breaking in QFT?   | The ground state fails to respect a symmetry of the laws, yielding Goldstone bosons (or masses via the Higgs mechanism).                 | Spontaneous symmetry breaking in quantum field theory occurs when the system's ground state does not share the symmetry of its underlying laws, causing certain symmetries to appear broken without explicit symmetry-breaking terms.                                 |  4.28432 |    0.296296 |   0.0769231 |       0.222222 |
| What is the Feynman propagator in QFT?          | The time‑ordered Green’s function giving the amplitude to travel between points; the inverse of (□+m²) with causal boundary conditions.  | The Feynman propagator in quantum field theory is the time-ordered Green's function describing the probability amplitude for a particle to travel between two spacetime points.                                                                                       |  7.26586 |    0.541667 |   0.304348  |       0.5      |
| What is a sufficient statistic?                 | A function of the data that retains all information about a parameter for a given model—no loss for inference.                           | A sufficient statistic is a function of the data that captures all information needed to estimate a parameter, making the original data unnecessary for inference about that parameter.                                                                               | 17.5924  |    0.583333 |   0.347826  |       0.5      |
| What is the central limit theorem?              | The mean/sum of many independent, finite‑variance variables is approximately normal.                                                     | The central limit theorem states that the sum or average of a large number of independent, identically distributed random variables tends to follow a normal distribution, regardless of the original variable's distribution.                                        |  2.98674 |    0.266667 |   0         |       0.266667 |
| What is the maximum likelihood estimator (MLE)? | The parameter value that maximizes the likelihood of the observed data under the model.                                                  | The maximum likelihood estimator is a statistical method that finds the parameter values maximizing the probability of observing the given data.                                                                                                                      |  3.07149 |    0.685714 |   0.181818  |       0.457143 |
| What is a confidence interval?                  | A procedure that covers the true parameter with a stated long‑run frequency (e.g., 95%); not a posterior probability.                    | A confidence interval is a range of values, derived from sample data, that likely contains the true population parameter with a specified level of confidence, such as 95%.                                                                                           |  7.03098 |    0.375    |   0.130435  |       0.333333 |
| What is a p-value?                              | The probability, assuming the null is true, of observing results at least as extreme as those seen.                                      | A p-value measures the probability of obtaining results at least as extreme as observed, assuming the null hypothesis is true, helping assess the evidence against the null hypothesis in statistical tests.                                                          | 25.188   |    0.612245 |   0.382979  |       0.367347 |
| What is VC dimension?                           | A capacity measure: the largest number of points a hypothesis class can shatter.                                                         | VC dimension measures a model's capacity by the largest set of points it can classify correctly in all possible ways, indicating its ability to fit diverse data patterns.                                                                                            |  4.17587 |    0.380952 |   0.1       |       0.333333 |
| What is empirical risk minimization (ERM)?      | Choose the model minimizing average loss on the training data.                                                                           | Empirical risk minimization (ERM) is a machine learning approach that selects a model by minimizing the average loss on a given training dataset, aiming to reduce prediction errors on observed data.                                                                |  5.72507 |    0.390244 |   0.102564  |       0.341463 |
| What is structural risk minimization (SRM)?     | Minimize a bound on risk by trading off fit and model complexity across nested classes (regularization).                                 | Structural risk minimization is a machine learning principle that balances model complexity and training error to minimize the overall risk of poor generalization on new data.                                                                                       |  3.79647 |    0.333333 |   0.05      |       0.190476 |
| What is the kernel trick?                       | Compute inner products via a kernel to act as if in high‑dimensional feature space without explicit mapping.                             | The kernel trick is a method in machine learning that allows algorithms to operate in high-dimensional spaces without explicitly computing coordinates, enabling efficient nonlinear classification and regression.                                                   |  2.01381 |    0.391304 |   0.136364  |       0.304348 |
| What is the bias–variance tradeoff?             | Reducing bias often increases variance; best generalization balances the two to minimize expected error.                                 | The bias–variance tradeoff is the balance between a model's simplicity (bias) and its sensitivity to data fluctuations (variance), affecting its accuracy and ability to generalize to new data.                                                                      |  1.62769 |    0.272727 |   0         |       0.181818 |
| What is hermeneutics?                           | The theory and methodology of interpretation, especially of texts.                                                                       | Hermeneutics is the study and theory of interpretation, especially of texts, language, and symbolic expressions, focusing on understanding meaning in context.                                                                                                        | 19.6963  |    0.533333 |   0.285714  |       0.466667 |
| What is semiotics?                              | The study of signs, symbols, and how meaning is created.                                                                                 | Semiotics is the study of signs and symbols and how they create meaning in communication and culture.                                                                                                                                                                 | 11.6439  |    0.740741 |   0.4       |       0.592593 |
| What is historiography?                         | The study of how history is written and the methods, sources, and interpretations historians use.                                        | Historiography is the study of how history is written, including the methods, sources, interpretations, and biases historians use to understand and present past events.                                                                                              | 32.7344  |    0.769231 |   0.486486  |       0.666667 |
| What is deconstruction?                         | A critical approach that exposes internal tensions in texts and shows meanings as contingent and unstable.                               | Deconstruction is a critical approach that analyzes texts to reveal hidden meanings, contradictions, and assumptions, showing how language shapes understanding and challenges fixed interpretations.                                                                 |  6.35205 |    0.45     |   0.157895  |       0.4      |
| What is cultural relativism?                    | The view that beliefs and practices should be understood within their cultural context rather than judged externally.                    | Cultural relativism is the idea that a person's beliefs and practices should be understood based on their own culture, not judged by the standards of another culture.                                                                                                | 18.0121  |    0.488889 |   0.232558  |       0.488889 |
| How to delete Facebook account?                 | Settings → Accounts Center → Personal details → Account ownership & control → Deactivation or deletion → Delete account → confirm.       | Go to Facebook settings, select "Your Facebook Information," then "Deactivation and Deletion," choose "Delete Account," follow prompts, and confirm to permanently delete your Facebook account.                                                                      |  1.55948 |    0.358974 |   0.0540541 |       0.307692 |
| How to delete Instagram account?                | Settings → Accounts Center → Personal details → Account ownership & control → Deactivation or deletion → Delete account → confirm.       | To delete your Instagram account, go to the Delete Your Account page on Instagram's website, log in, select a reason, re-enter your password, and click "Permanently delete my account." This cannot be undone.                                                       |  1.23504 |    0.204082 |   0         |       0.163265 |
| How to screenshot on Mac?                       | Shift+⌘+3 (full), Shift+⌘+4 (area), Shift+⌘+5 (options/record).                                                                          | Press Command + Shift + 3 to capture the entire screen, or Command + Shift + 4 to select a portion to screenshot on a Mac.                                                                                                                                            |  5.44971 |    0.25     |   0.133333  |       0.25     |
| How to screenshot on Windows?                   | Win+Shift+S (Snipping Tool), PrtScn (copy screen), Win+PrtScn (save to Pictures/Screenshots).                                            | Press the "PrtScn" key to copy the screen to the clipboard, then paste it into an app. Or press "Windows + Shift + S" to select and capture part of the screen.                                                                                                       |  7.42506 |    0.272727 |   0.047619  |       0.181818 |
| How to tie a tie?                               | For a four‑in‑hand: cross wide over narrow, up through loop, around front, up through front loop, pull down and center.                  | Place the tie around your neck, cross the wide end over the narrow end, loop it under and through the neck loop, then bring it down through the front knot and tighten.                                                                                               |  3.44879 |    0.444444 |   0         |       0.333333 |
| How long to boil eggs?                          | After boiling starts: soft 6–7 min, medium 8–9, hard 10–12; then ice bath.                                                               | Boil eggs for about 9-12 minutes for hard-boiled, 6-7 minutes for medium, and 4-5 minutes for soft-boiled. Adjust time based on egg size and altitude.                                                                                                                |  1.49337 |    0.347826 |   0.0454545 |       0.173913 |
| How to calculate percentage?                    | Percentage = (part/whole) × 100; e.g., 25 of 200 → 12.5%.                                                                                | To calculate a percentage, divide the part by the whole, then multiply the result by 100. For example, (part ÷ whole) × 100 = percentage.                                                                                                                             |  9.81216 |    0.242424 |   0.129032  |       0.242424 |
| How to lose weight fast?                        | Safest is gradual: 0.5–1% body weight/week via calorie deficit, high protein, resistance training, sleep; avoid crash diets.             | To lose weight fast, reduce calorie intake, eat nutritious whole foods, increase physical activity, stay hydrated, and get enough sleep. Consult a healthcare professional before starting any rapid weight loss plan.                                                |  1.57631 |    0.117647 |   0         |       0.117647 |
| How to lower blood pressure?                    | DASH‑style eating, less sodium (~1.5–2 g/day), regular exercise, limit alcohol, healthy weight, stress control; follow clinician advice. | To lower blood pressure, eat a healthy diet low in salt, exercise regularly, maintain a healthy weight, reduce alcohol intake, manage stress, avoid smoking, and follow your doctor's advice and medications if prescribed.                                           |  4.80909 |    0.290909 |   0.0377358 |       0.254545 |
| How to make money online?                       | Sell skills (freelance), products (e‑commerce), or content; validate demand and avoid “get‑rich‑quick” schemes.                          | You can make money online by freelancing, selling products, affiliate marketing, creating content, tutoring, or offering services through platforms like Upwork, Etsy, Amazon, or YouTube.                                                                            |  2.8651  |    0.243902 |   0         |       0.146341 |


## Підсумкові середні значення метрик

| Метрика | Середнє значення |
|---------|-------------------|
| BLEU (avg) | 7.445 |
| ROUGE-1 F1 (avg) | 0.398 |
| ROUGE-2 F1 (avg) | 0.137 |
| ROUGE-L F1 (avg) | 0.320 |


---

## Інтерпретація метрик за категоріями

За науковими питаннями (**Quantum Field Theory, Statistics, Statistical Learning**) метрики показали відносно низькі BLEU, але помірні значення ROUGE-1 та ROUGE-L. Це свідчить про те, що модель OpenAI часто відповідала коректно за змістом, але з використанням інших формулювань, ніж у референсах. Особливо в QFT та Statistical Learning ROUGE-2 залишався низьким, що вказує на відмінності у точних біграмах та термінах, тоді як у Statistics результати були дещо кращими завдяки усталеним дефініціям (наприклад, MLE чи p-value).

У категоріях **Humanities** результати значно вищі: ROUGE-1 і ROUGE-L показали добрий збіг (0.5–0.7), а BLEU часто був високим завдяки стабільним визначенням понять на кшталт *historiography* чи *semiotics*. Натомість **Everyday** питання продемонстрували найгірші показники, зокрема дуже низькі BLEU і ROUGE-2, оскільки еталонні відповіді були надзвичайно стислими інструкціями, а модель повертала довші та описові тексти. Таким чином, гуманітарні дефініції краще узгоджуються з еталоном, тоді як прикладні інструкції суттєво знижують метрики.


---

## Підсумкові спостереження
- Високі значення для стислих дефініцій зі стабільною термінологією (*historiography, p-value, sufficient statistic, semiotics*).
- Низькі — коли відповідь розлого перефразована або має іншу структуру кроків (*how to delete…*, *screenshots*, *tie a tie*).
- Суворе обмеження довжини та інструкція «1 речення, без прелюдій» підвищують ROUGE-2 і BLEU.

