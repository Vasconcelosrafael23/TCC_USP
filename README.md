# Modelos estatísticos e de aprendizado de máquina na previsão de tráfego em rodovias concedidas

Código do Trabalho de Conclusão de Curso do MBA em Data Science e Analytics da USP/Esalq.

**Autor:** Rafael Vasconcelos
**Orientador:** José Erasmo Silva

## Sobre o trabalho

O trabalho compara quatro modelos de previsão do volume mensal de tráfego de uma praça de pedágio de rodovia federal concedida (Vargem/SP, BR-381), com dados de janeiro de 2010 a novembro de 2024:

- Holt-Winters aditivo;
- SARIMA;
- Prophet;
- XGBoost.

Os quatro são confrontados com um modelo de referência, o naïve sazonal, que repete o volume do mesmo mês do ano anterior.

A comparação usa validação por origem móvel nos horizontes de um e de doze meses, as métricas MAE, RMSE, MAPE e MASE e o teste de Diebold-Mariano com correção de Holm. O trabalho também examina o procedimento de projeção de demanda adotado na estruturação de concessões, por análise documental e por um teste do método dos estudos na mesma janela de avaliação dos modelos.

## Dados

Os dados vêm do conjunto [Volume de Tráfego nas Praças de Pedágio](https://dados.antt.gov.br/dataset/volume-trafego-praca-pedagio), do Portal de Dados Abertos da Agência Nacional de Transportes Terrestres (ANTT).

Os dados não estão neste repositório. Na primeira execução, o script cria a pasta `dados/` e baixa automaticamente do portal da ANTT os arquivos anuais de 2010 a 2024, que bastam para todos os resultados.

Como a ANTT pode alterar os endereços ou o conteúdo dos arquivos, a reprodução exata dos números depende de os dados publicados permanecerem os mesmos.

## Como executar

Testado com Python 3.12.3.

```bash
pip install -r requirements.txt
python tcc_previsao_trafego.py
```

A execução completa leva cerca de 10 minutos.

## O que o programa faz

1. Lê os arquivos da ANTT e monta a série mensal da praça (179 meses, sem lacunas).
2. Decompõe a série pela STL e calcula as forças de tendência e de sazonalidade.
3. Aplica os testes de estacionariedade ADF, KPSS, OCSB e Canova-Hansen.
4. Escolhe a configuração do Prophet usando apenas o treino (validação interna nos últimos 24 meses do treino).
5. Ajusta os cinco modelos e os valida por origem móvel, com 24 meses de teste.
6. Calcula as métricas de erro e o teste de Diebold-Mariano, com correção de Holm para comparações múltiplas.
7. Aplica aos erros do melhor modelo o teste de Ljung-Box e o teste t de média nula.
8. Repete a validação no horizonte de doze meses e com o SARIMA sem diferença sazonal.
9. Testa a sensibilidade à quebra da pandemia, com março a dezembro de 2020 ajustados.
10. Projeta os doze meses seguintes, com intervalo de 95% para o Holt-Winters.
11. Calcula o efeito ilustrativo de um viés de 1 ponto percentual na taxa de crescimento sobre a tarifa.
12. Aplica o método dos estudos de concessão (crescimento esperado do PIB, pelo Focus de cada data, multiplicado por elasticidade unitária) a partir de nov/2022 e de nov/2023 e o compara com as previsões de doze meses dos modelos nas mesmas origens.

## Arquivos gerados

| Arquivo | Conteúdo |
| --- | --- |
| `resultados_para_texto.txt` | Relatório da rodada principal |
| `selecao_prophet.txt` | Escolha da configuração do Prophet na validação interna do treino |
| `exercicio_metodo_estudos.txt` | Comparação do método dos estudos com os modelos (Tabela 6 do TCC) |
| `analises_complementares.txt` | Correção de Holm, testes sazonais, SARIMA com D = 0 e viés de 1 p.p. |
| `numeros_do_texto.txt` | Números citados no texto do TCC que não constam das tabelas |
| `serie_prevista.xlsx` | Série histórica, previsões do teste, previsão futura e métricas |
| `metricas_h1.csv`, `metricas_h12.csv` | Métricas por modelo nos horizontes de um e de doze meses |
| `figura1_stl.png`, `figura2_janela_teste.png`, `figura3_erro_por_passo.png`, `figura4_previsao.png` | Figuras do TCC |

## Ambiente

Os resultados foram reproduzidos com as versões listadas em `requirements.txt`. Os modelos usam semente fixa (42), de modo que a execução reproduz os números do trabalho.
