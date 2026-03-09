
Overview
The Epigenetic Clock Challenge
What is an Epigenetic Clock?

DNA methylation is a biochemical process in which a methyl group is added to cytosine bases in DNA, predominantly at CpG dinucleotide sites. These modifications play a crucial role in gene regulation without altering the underlying DNA sequence. Remarkably, DNA methylation patterns change in a predictable manner as we age, acting as a biological clock — the so-called epigenetic clock.

Pioneering work by Steve Horvath (2013) and Gregory Hannum et al. demonstrated that a relatively small subset of CpG sites can be used to predict an individual's chronological age with striking accuracy. The difference between predicted and actual age — known as epigenetic age acceleration — has since been linked to various health outcomes, mortality risk, and disease susceptibility.
Biological Context

This challenge is inspired by real-world data from the study GSE42861 (Liu & Feinberg, 2013), which profiled genome-wide DNA methylation in peripheral blood leukocytes using the Illumina HumanMethylation450 BeadChip. This array measures methylation levels at over 450,000 CpG sites across the human genome, providing a rich molecular portrait of each individual.

In the original study, DNA from blood samples was subjected to bisulfite conversion — a chemical treatment that converts unmethylated cytosines to uracil while leaving methylated cytosines unchanged — followed by hybridization to the array. The resulting methylation values (beta values, ranging from 0 to 1) quantify the proportion of methylated molecules at each CpG site.
The Challenge

You are provided with two datasets ([X_train, y_train] and X_test) containing:

    DNA methylation profiles measured across a subset of CpG sites from blood samples
    Chronological age of each individual (available only in the training set)

Your objective: build a predictive model in Python that learns the relationship between CpG methylation patterns and age from the training data, and accurately predicts the age of individuals in the test set (i.e. y_test).
Why Does This Matter?

Epigenetic clocks have become a cornerstone of aging research. Accurate age prediction from methylation data enables:

    Forensic applications — estimating the age of individuals from biological samples
    Clinical research — identifying patients who are aging faster or slower than expected (epigenetic age acceleration), which correlates with disease risk
    Drug development — evaluating the impact of therapeutic interventions on biological aging
    Public health — understanding the epigenetic effects of lifestyle, environment, and socioeconomic factors on aging trajectories

Suggested Approaches

This is fundamentally a regression problem. You are encouraged to explore various statistical and machine learning methods, such as:

    Penalized regression (Ridge, Lasso, Elastic Net)
    Support Vector Regression
    Random Forests / Gradient Boosted Trees
    Neural Networks
    Dimensionality reduction combined with regression (PCA + regression)

A key difficulty is that the number of features (CpG sites) is much larger than the number of samples — a classic high-dimensional setting (p >> n) — making regularization and feature selection critical.
Acknowledgements

The Epiclock challenge was originally created by Florent Chuffart and is available at https://github.com/fchuffar/starting_kit_epiclock1.0. His work was supported by the RIS (Réseau Inter-disciplinaire autour de la Statistique).
