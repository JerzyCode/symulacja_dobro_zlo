## Modelowanie równowagi dobra i zła w układach złożonych

### 2.1 Równanie replikatora (Replicator Equation)

W populacji są dwie strategie:
- C - dobro (współpraca)
- D - zło (egoizm)

Niech x - odsetek kooperatroów $(0 \leq x \leq 1)$

## Średni zysk (payoff)


Każda strategia ma średni zysk, który zależy od tego, kogo spotykają osobniki w populacji.

$$
\pi_C = R \cdot x + S \cdot (1-x)
$$

$$
\pi_D = T \cdot x + P \cdot (1-x)
$$

Gdzie:

- **R** = nagroda za współpracę z kooperatorem (C-C)  
- **S** = strata / „słaba korzyść” dla kooperatora, który spotyka defektora (C-D)  
- **T** = zysk defektora przy spotkaniu kooperatora (D-C, *temptation*)  
- **P** = zysk defektora przy spotkaniu innego defektora (D-D, *punishment*)  
- **PI_C** - średni zysk kooperatorów
- **PI_D** - średni zysk defektoróœ

**Interpretacja:**

- Jeśli większość populacji współpracuje (x bliskie 1), kooperatorzy mają wysoki zysk:  
  $$
  \pi_C = R \cdot x + S \cdot (1-x)
  $$  
  bo często spotykają innych kooperatorów.

- Jeśli większość populacji to defektorzy (x bliskie 0), defektorzy mają przewagę:  
  $$
  \pi_D = T \cdot x + P \cdot (1-x)
  $$  

  bo wykorzystują kooperatorów.


Średni wynik:

$$
\bar{\pi} = x \pi_c + (1-x) \pi_d
$$

### Równanie defektor

$$
\frac{dx}{dt} = x (\pi_C - \bar{\pi})
$$

### Rozwiązanie

$$
-(R - S - t + P)x^2 + (R - 2S - t + 2P)x + (S - P) = 0
$$