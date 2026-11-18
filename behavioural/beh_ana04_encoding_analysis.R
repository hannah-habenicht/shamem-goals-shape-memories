# ==================================================================
# SHAMEM 1: Ana 04 ENCODING ANALYSIS
# ==================================================================
# RT outliers exclude corresponding Accuracy trials & absolute RT cutoff (>2.5s)
# Participant outliers: Acc (Exclude), RT (Flag only).
# Accuracy Calculation: Sum(Correct_Kept) / Fixed_Design_Trials.
# Dynamic Denominators: Handles variable trials per film (15 vs 12 coins per film).
# ==================================================================

rm(list = ls())
set.seed(1)

# ---- Paths: edit these two lines ----
DATA_ROOT   <- "path/to/data_deposit"   # unpacked data deposit
RESULTS_DIR <- "path/to/results"        # e.g. the repository's results/ folder

PATH_IN  <- file.path(DATA_ROOT, "behavioural_results")    # shamem_encoding_behav.csv
PATH_OUT <- file.path(RESULTS_DIR, "behavioural")
dir.create(PATH_OUT, recursive = TRUE, showWarnings = FALSE)

library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)
library(scales)
library(tidyr)
library(car)
library(dplyr)


options(contrasts = c("contr.sum", "contr.poly"))

setwd(PATH_IN)

number_categories <- 3
number_blocks <- 3

##### 1. Load & prepare Data ####
df <- read.csv("shamem_encoding_behav.csv")

# Group assignment from the corrected conditions.csv (type = timing), replacing
# the labels in shamem_encoding_behav.csv, which come from the raw file names
# (they differ for participant 14 only).
conditions <- read.csv(file.path(DATA_ROOT, "study_info", "conditions.csv")) %>%
  dplyr::select(id, timing = type, job)
df <- df %>%
  dplyr::select(-timing, -job) %>%
  dplyr::left_join(conditions, by = "id")
stopifnot(!anyNA(df$timing), !anyNA(df$job))

df <- df %>%
  rename(
    Participant_ID = "id",
    film           = "filmID",
    block          = "block",
    trialID        = "trialID",
    job            = "job",
    timing         = "timing",
    category       = "category",
    DV_Value       = "output"
  ) %>%
  mutate(
    Participant_ID = factor(Participant_ID),
    film           = factor(film),
    session        = factor(session),
    block          = factor(block, levels = c(1, 2, 3)),
    job = factor(job, levels = c(1, 2, 3), labels = c("Spy", "Criminal", "Neutral")),
    timing = factor(timing, levels = c(1, 2, 3), labels = c("Pro", "Retro", "Neutral")),
    category = factor(category, levels = c(3, 1, 2), labels = c("Object", "Face", "Place"))
  )

# --- DEFINE TRIAL COUNTS PER FILM ---
# Films 1-3: 15 trials/coins total (5 per category)
# Films 4-6: 12 trials/coins total (4 per category)
film_trial_counts <- tibble(
  film = factor(c(1, 2, 3, 4, 5, 6)), 
  Trials_Per_Cat = c(5, 5, 5, 4, 4, 4)
)
#### end ####


setwd(PATH_OUT)

##### 2. TRIAL-LEVEL CLEANING (Z-SCORE + ABSOLUTE CUTOFF) ####

# Step 2a: Z-Score Cleaning (computed WITHIN measure AND within category)
df_clean_trials <- df %>%
  filter(!is.na(DV_Value)) %>%
  group_by(Participant_ID, measure, category, block) %>%
  mutate(
    Cond_Mean = mean(DV_Value, na.rm = TRUE),
    Cond_SD   = sd(DV_Value, na.rm = TRUE),
    Z_Score   = (DV_Value - Cond_Mean) / Cond_SD,
    Keep = case_when(
      measure == "RT" ~ abs(Z_Score) < 3,   # only RT gets the z-score rule
      TRUE ~ TRUE                            # accuracy always kept here
    )
  ) %>%
  ungroup()

cat("Trials removed by Z-Score (>3 SD):", sum(!df_clean_trials$Keep), "\n")

# Step 2b: Absolute RT Cutoff (> 2.5 seconds)
cutoff_threshold <- 2.5 

df_clean_trials <- df_clean_trials %>%
  mutate(
    Keep = case_when(
      measure == "RT" & DV_Value > cutoff_threshold ~ FALSE,
      TRUE ~ Keep
    )
  )

cat("Total trials removed (Z-Score + >2.5s RT cutoff):", sum(!df_clean_trials$Keep), "\n")

# Step 2c: Cross-Measure Exclusion Logic
failed_key <- df_clean_trials %>%
  filter(measure == "RT" & !Keep) %>%
  distinct(Participant_ID, film, block, trialID)

df_clean_trials <- df_clean_trials %>%
  anti_join(failed_key, by = c("Participant_ID", "film", "block", "trialID")) %>%
  mutate(Keep = ifelse(measure == "RT", Keep, Keep))  # keep semantics unchanged for RT rows

cat("Final trial count after syncing across measures:", sum(df_clean_trials$Keep), "\n")

# PLOTTING TRIAL OUTLIERS #
plot_data_vis <- df_clean_trials %>%
  filter(measure == "RT")

p_clean_check <- ggplot(plot_data_vis, aes(x = interaction(job, timing, category), y = DV_Value, color = factor(Keep))) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 1) +
  facet_wrap(~ block, scales = "free_y") +
  scale_color_manual(values = c("TRUE" = "gray60", "FALSE" = "red"), labels = c("Kept", "Removed")) +
  labs(title = "Outlier Removal Check (Encoding RT)",
       x = "Condition", y = "Reaction Time (s)", color = "Status") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom")

print(p_clean_check)
ggsave(filename = "Encoding_trialcleaning.png", plot = p_clean_check, path = PATH_OUT, dpi = 300)
#### end ####

##### 3. PARTICIPANT-LEVEL OUTLIER DETECTION #####

# --- A. ACCURACY OUTLIERS (EXCLUDE GLOBALLY) ---
# Calculate Mean Accuracy per Participant per Block using actual trial counts
df_subj_acc <- df_clean_trials %>%
  filter(Keep & measure == "acc") %>%
  group_by(Participant_ID, block) %>%
  summarise(
    Sum_Correct = sum(DV_Value, na.rm = TRUE),
    # Denominator: Sum of (Trials_Per_Cat * 3 categories) for all films in this block
    # Since every block has all 6 films: (15*3) + (15*3) + (15*3) + (12*3) + (12*3) + (12*3) = 243 total trials per block
    Total_Trials_Design = sum(film_trial_counts$Trials_Per_Cat * number_categories), 
    Mean_Acc = Sum_Correct / Total_Trials_Design,
    .groups = "drop"
  )


df_subj_acc_outliers <- df_subj_acc %>%
  group_by(block) %>%
  mutate(
    Group_Mean = mean(Mean_Acc, na.rm = TRUE),
    Group_SD   = sd(Mean_Acc, na.rm = TRUE),
    Subj_Z     = (Mean_Acc - Group_Mean) / Group_SD,
    Is_Outlier = abs(Subj_Z) > 3
  ) %>%
  ungroup()

ids_exclude_acc <- df_subj_acc_outliers %>%
  filter(Is_Outlier) %>%
  distinct(Participant_ID)

if(nrow(ids_exclude_acc) > 0) {
  cat("\n--- PARTICIPANTS EXCLUDED (Low Accuracy) ---\n")
  print(ids_exclude_acc)
  p_acc_outliers <- ggplot(df_subj_acc_outliers, aes(x = factor(block), y = Mean_Acc, color = Is_Outlier)) +
    geom_point(size = 3, alpha = 0.7) +
    scale_color_manual(values = c("FALSE" = "blue", "TRUE" = "red"), labels = c("Normal", "Excluded")) +
    labs(title = "Participant Outliers: Accuracy (Excluded)", x = "Block", y = "Mean Accuracy") +
    theme_minimal()
  print(p_acc_outliers)
  ggsave(filename = "Encoding_accoutliers.png", plot = p_acc_outliers, path = PATH_OUT, dpi = 300)
  
} else {
  cat("\nNo Accuracy outliers detected.\n")
}

# --- B. RT OUTLIERS (FLAG ONLY) ---
df_subj_rt <- df_clean_trials %>%
  filter(Keep & measure == "RT") %>%
  group_by(Participant_ID, block) %>%
  summarise(Mean_RT = mean(DV_Value, na.rm = TRUE), .groups = "drop")

df_subj_rt_outliers <- df_subj_rt %>%
  group_by(block) %>%
  mutate(
    Group_Mean = mean(Mean_RT, na.rm = TRUE),
    Group_SD   = sd(Mean_RT, na.rm = TRUE),
    Subj_Z     = (Mean_RT - Group_Mean) / Group_SD,
    Is_Outlier = abs(Subj_Z) > 3
  ) %>%
  ungroup()

ids_flag_rt <- df_subj_rt_outliers %>%
  filter(Is_Outlier) %>%
  distinct(Participant_ID)

if(nrow(ids_flag_rt) > 0) {
  cat("\n--- PARTICIPANTS FLAGGED (Slow RT) - NOT EXCLUDED ---\n")
  print(ids_flag_rt)
  p_rt_outliers <- ggplot(df_subj_rt_outliers, aes(x = factor(block), y = Mean_RT, color = Is_Outlier)) +
    geom_point(size = 3, alpha = 0.7) +
    scale_color_manual(values = c("FALSE" = "blue", "TRUE" = "orange"), labels = c("Normal", "Flagged")) +
    labs(title = "Participant Outliers: RT (Retained)", x = "Block", y = "Mean RT (s)") +
    theme_minimal()
  print(p_rt_outliers)
  ggsave(filename = "Encoding_RToutliers.png", plot = p_rt_outliers, path = PATH_OUT, dpi = 300)
  
} else {
  cat("\nNo RT outliers flagged.\n")
}

# --- C. CREATE FINAL DATASET ---
ids_to_remove <- unique(ids_exclude_acc$Participant_ID)

df_final_analysis <- df_clean_trials %>%
  filter(Keep) %>%
  filter(!Participant_ID %in% ids_to_remove)

cat("\nTotal Participants Excluded (Accuracy):", length(ids_to_remove), "\n")
cat("Final dataset rows:", nrow(df_final_analysis), "\n")
#### end ####

#### 4. OVERALL PERFORMANCE PLOTTING ####
cat("\n=== OVERALL PERFORMANCE PLOTTING ===\n")

df_overall <- df_final_analysis %>%
  mutate(
    Group_Enc = case_when(
      timing == "Pro" & job == "Spy" ~ "Spy_Pro",
      timing == "Pro" & job == "Criminal" ~ "Crim_Pro",
      TRUE ~ "Control"
    ),
    Group_Enc = factor(Group_Enc, levels = c("Control", "Spy_Pro", "Crim_Pro"))
  )


# Aggregate: Sum Correct (Numerator) and Count Design Trials (Denominator)
df_stats_clean <- bind_rows(
  df_overall %>% filter(measure == "acc") %>%
    group_by(Participant_ID, block, Group_Enc, category) %>%
    summarise(DV_Value = sum(DV_Value, na.rm = TRUE) /
                sum(film_trial_counts$Trials_Per_Cat), .groups = "drop") %>%
    mutate(measure = "acc"),
  df_overall %>% filter(measure == "RT") %>%
    group_by(Participant_ID, block, Group_Enc, category) %>%
    summarise(DV_Value = mean(DV_Value, na.rm = TRUE), .groups = "drop") %>%
    mutate(measure = "RT")
)

# Prepare for Plotting
plot_data_overall <- df_stats_clean %>%
  group_by(block, Group_Enc, measure) %>%
  summarise(Mean_Val = mean(DV_Value, na.rm = TRUE),
            SD_Val = sd(DV_Value, na.rm = TRUE),
            N = n(),
            SE_Val = SD_Val / sqrt(N), .groups = "drop")

df_subj_means_block <- df_stats_clean %>%
  group_by(Participant_ID, block, Group_Enc, measure) %>%
  summarise(Subj_Mean = mean(DV_Value, na.rm = TRUE), .groups = "drop")

df_subj_means_category <- df_stats_clean %>%
  group_by(Participant_ID, category, Group_Enc, measure) %>%
  summarise(Subj_Mean = mean(DV_Value, na.rm = TRUE), .groups = "drop")

# Sanity Check
cat("\n--- Sanity Check: Min Subject Accuracy per Category ---\n")
check_mins <- df_stats_clean %>%
  filter(measure == "acc") %>%
  group_by(Participant_ID, category) %>%
  summarise(Subj_Mean_Cat = mean(DV_Value, na.rm = TRUE), .groups = "drop") %>%
  group_by(category) %>%
  summarise(Min_Acc = min(Subj_Mean_Cat), Max_Acc = max(Subj_Mean_Cat), .groups = "drop")
print(check_mins)

# Plots (Blocks)
p_acc_overall <- ggplot() +
  geom_point(data = df_subj_means_block %>% filter(measure == "acc"),
             aes(x = block, y = Subj_Mean, color = Group_Enc, group = Group_Enc),
             position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.2),
             alpha = 0.3, size = 1.5) +
  geom_line(data = plot_data_overall %>% filter(measure == "acc"),
            aes(x = block, y = Mean_Val, color = Group_Enc, group = Group_Enc), size = 1.2) +
  geom_point(data = plot_data_overall %>% filter(measure == "acc"),
             aes(x = block, y = Mean_Val, color = Group_Enc, group = Group_Enc), size = 3, shape = 18) +
  geom_errorbar(data = plot_data_overall %>% filter(measure == "acc"),
                aes(x = block, ymin = Mean_Val - SE_Val, ymax = Mean_Val + SE_Val, color = Group_Enc),
                width = 0.1, size = 0.8) +
  labs(title = "Overall Accuracy (Correct Kept / Design Total)", y = "Accuracy", x = "Block") +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  theme_minimal() + theme(legend.position = "bottom")
print(p_acc_overall)
ggsave(filename = "Encoding_raw_acc.png", plot = p_acc_overall, path = PATH_OUT, dpi = 300)

p_rt_overall <- ggplot() +
  geom_point(data = df_subj_means_block %>% filter(measure == "RT"),
             aes(x = block, y = Subj_Mean, color = Group_Enc, group = Group_Enc),
             position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.2),
             alpha = 0.3, size = 1.5) +
  geom_line(data = plot_data_overall %>% filter(measure == "RT"),
            aes(x = block, y = Mean_Val, color = Group_Enc, group = Group_Enc), size = 1.2) +
  geom_point(data = plot_data_overall %>% filter(measure == "RT"),
             aes(x = block, y = Mean_Val, color = Group_Enc, group = Group_Enc), size = 3, shape = 18) +
  geom_errorbar(data = plot_data_overall %>% filter(measure == "RT"),
                aes(x = block, ymin = Mean_Val - SE_Val, ymax = Mean_Val + SE_Val, color = Group_Enc),
                width = 0.1, size = 0.8) +
  labs(title = "Overall RT (Kept Trials Mean)", y = "RT (s)", x = "Block") +
  theme_minimal() + theme(legend.position = "bottom")
print(p_rt_overall)
ggsave(filename = "Encoding_raw_RT.png", plot = p_rt_overall, path = PATH_OUT, dpi = 300)

# Plots (Category - Violins)
p_acc_cat <- ggplot() +
  geom_violin(data = df_subj_means_category %>% filter(measure == "acc"),
              aes(x = category, y = Subj_Mean, fill = category),
              trim = FALSE, alpha = 0.6, color = "black", scale = "width") +
  geom_point(data = df_subj_means_category %>% filter(measure == "acc"),
             aes(x = category, y = Subj_Mean, color = Group_Enc, group = Group_Enc),
             position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.2),
             alpha = 0.3, size = 1.5) +
  #              width = 0.15, alpha = 0.5, size = 1.5, color = "black", height = 0) +
  stat_summary(data = df_subj_means_category %>% filter(measure == "acc"),
               aes(x = category, y = Subj_Mean), fun = "mean", geom = "point", shape = 18, size = 4, color = "white") +
  labs(title = "Accuracy by Category", subtitle = "Denominator adjusts per Film", y = "Accuracy", x = "Category") +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  theme_minimal() + theme(legend.position = "none")
print(p_acc_cat)
ggsave(filename = "Encoding_rawcat_acc.png", plot = p_acc_cat, path = PATH_OUT, dpi = 300)

p_rt_cat <- ggplot() +
  geom_violin(data = df_subj_means_category %>% filter(measure == "RT"),
              aes(x = category, y = Subj_Mean, fill = category),
              trim = FALSE, alpha = 0.6, color = "black", scale = "width") +
  geom_point(data = df_subj_means_category %>% filter(measure == "RT"),
             aes(x = category, y = Subj_Mean, color = Group_Enc, group = Group_Enc),
             position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.2),
             alpha = 0.3, size = 1.5) +
  stat_summary(data = df_subj_means_category %>% filter(measure == "RT"),
               aes(x = category, y = Subj_Mean), fun = "mean", geom = "point", shape = 18, size = 4, color = "white") +
  labs(title = "RT by Category", y = "RT (s)", x = "Category") +
  theme_minimal() + theme(legend.position = "none")
print(p_rt_cat)
ggsave(filename = "Encoding_rawcat_RT.png", plot = p_rt_cat, path = PATH_OUT, dpi = 300)
#### end ####

#### 5. DESCRIPTIVE STATISTICS FOR REPORTING ####
# Calculates Mean and SE for all requested levels.
# Uses the same data sources (df_stats_clean, df_subj_means_*) as the plots.

cat("\n=================================================================\n")
cat("DESCRIPTIVE STATISTICS FOR REPORTING\n")
cat("=================================================================\n")

# Helper function to print a nice table
print_table <- function(df, title_name) {
  cat("\n---", title_name, "---\n")
  # Format numbers: 3 decimals for RT, 2 for Accuracy (handled in print)
  print(df, row.names = FALSE)
  cat("\n")
}


# PART A: ACCURACY DESCRIPTIVES
cat("\n=== ACCURACY (Proportion Correct) ===\n")

# 1. Accuracy by Block (Across everything)
# Source: plot_data_overall (Group x Block) -> We collapse Group to get "Across Everything"
acc_block <- df_stats_clean %>%
  filter(measure == "acc") %>%
  group_by(block) %>%
  summarise(
    Mean = mean(DV_Value, na.rm = TRUE),
    SE = sd(DV_Value, na.rm = TRUE) / sqrt(n()),
    N = n(),
    .groups = "drop"
  ) %>%
  mutate(Mean = round(Mean, 3), SE = round(SE, 3))

print_table(acc_block, "Accuracy: Overall by Block (1, 2, 3)")

# 2. Accuracy by Category (Across everything)
acc_cat <- df_stats_clean %>%
  filter(measure == "acc") %>%
  group_by(category) %>%
  summarise(
    Mean = mean(DV_Value, na.rm = TRUE),
    SE = sd(DV_Value, na.rm = TRUE) / sqrt(n()),
    N = n(),
    .groups = "drop"
  ) %>%
  mutate(Mean = round(Mean, 3), SE = round(SE, 3))

print_table(acc_cat, "Accuracy: Overall by Category")

# 3. Accuracy by Group & Category (Face/Place for Control, Spy, Crim)
# This matches your specific hypothesis groups
acc_group_cat <- df_stats_clean %>%
  filter(measure == "acc", category %in% c("Face", "Place")) %>%
  group_by(Group_Enc, category) %>%
  summarise(
    Mean = mean(DV_Value, na.rm = TRUE),
    SE = sd(DV_Value, na.rm = TRUE) / sqrt(n()),
    N = n(),
    .groups = "drop"
  ) %>%
  # Pivot to make it easier to read (Face vs Place side-by-side)
  pivot_wider(names_from = category, values_from = c(Mean, SE, N)) %>%
  mutate(
    across(starts_with("Mean"), ~round(.x, 3)),
    across(starts_with("SE"), ~round(.x, 3))
  )

print_table(acc_group_cat, "Accuracy: Mean (SE) by Group and Category (Face vs Place)")


# PART B: REACTION TIME DESCRIPTIVES
cat("\n=== REACTION TIME (Seconds) ===\n")

# 1. RT by Block (Across everything)
rt_block <- df_stats_clean %>%
  filter(measure == "RT") %>%
  group_by(block) %>%
  summarise(
    Mean = mean(DV_Value, na.rm = TRUE),
    SE = sd(DV_Value, na.rm = TRUE) / sqrt(n()),
    N = n(),
    .groups = "drop"
  ) %>%
  mutate(Mean = round(Mean, 3), SE = round(SE, 3))

print_table(rt_block, "RT: Overall by Block (1, 2, 3)")

# 2. RT by Category (Across everything)
rt_cat <- df_stats_clean %>%
  filter(measure == "RT") %>%
  group_by(category) %>%
  summarise(
    Mean = mean(DV_Value, na.rm = TRUE),
    SE = sd(DV_Value, na.rm = TRUE) / sqrt(n()),
    N = n(),
    .groups = "drop"
  ) %>%
  mutate(Mean = round(Mean, 3), SE = round(SE, 3))

print_table(rt_cat, "RT: Overall by Category")

# 3. RT by Block AND Category
rt_block_cat <- df_stats_clean %>%
  filter(measure == "RT") %>%
  group_by(block, category) %>%
  summarise(
    Mean = mean(DV_Value, na.rm = TRUE),
    SE = sd(DV_Value, na.rm = TRUE) / sqrt(n()),
    N = n(),
    .groups = "drop"
  ) %>%
  mutate(Mean = round(Mean, 3), SE = round(SE, 3))

print_table(rt_block_cat, "RT: by Block and Category")

# 4. RT by Block AND Group
rt_block_grp <- df_stats_clean %>%
  filter(measure == "RT") %>%
  group_by(block, Group_Enc) %>%
  summarise(
    Mean = mean(DV_Value, na.rm = TRUE),
    SE = sd(DV_Value, na.rm = TRUE) / sqrt(n()),
    N = n(),
    .groups = "drop"
  ) %>%
  mutate(Mean = round(Mean, 3), SE = round(SE, 3))

print_table(rt_block_grp, "RT: by Block and Group")

# 5. RT by Group AND Category (Face/Place for Control, Spy, Crim)
rt_group_cat <- df_stats_clean %>%
  filter(measure == "RT", category %in% c("Face", "Place")) %>%
  group_by(Group_Enc, category) %>%
  summarise(
    Mean = mean(DV_Value, na.rm = TRUE),
    SE = sd(DV_Value, na.rm = TRUE) / sqrt(n()),
    N = n(),
    .groups = "drop"
  ) %>%
  # Pivot for side-by-side comparison
  pivot_wider(names_from = category, values_from = c(Mean, SE, N)) %>%
  mutate(
    across(starts_with("Mean"), ~round(.x, 3)),
    across(starts_with("SE"), ~round(.x, 3))
  )

print_table(rt_group_cat, "RT: Mean (SE) by Group and Category (Face vs Place)")

#### end ####

#### 6. ACCURACY ANALYSIS: GLMM (BINOMIAL) ####

# --- 1. Prepare Data ---
data_acc_glmm <- df_final_analysis %>%
  filter(measure == "acc") %>%
  mutate(
    Group_Enc = case_when(
      timing == "Pro" & job == "Spy" ~ "Spy_Pro",
      timing == "Pro" & job == "Criminal" ~ "Crim_Pro",
      TRUE ~ "Control"
    ),
    Group_Enc = factor(Group_Enc, levels = c("Control", "Spy_Pro", "Crim_Pro")),
    # Ensure outcome is numeric 0/1
    DV_Value = as.numeric(DV_Value)
  )

cat("Running Accuracy GLMM on", nrow(data_acc_glmm), "trials...\n")

# --- 2. Run Model ---
# not stable versions:
#model_acc_glmm <- glmer(
#  DV_Value ~ Group_Enc * block * category + film +
#    (1 + category + block| Participant_ID) +
#    (1 | film),
#  data = data_acc_glmm,
#  family = binomial(link = "logit")
#)

#model_acc_glmm <- glmer(# <- was not stable
#  DV_Value ~ Group_Enc * block * category + film +
#    (1| Participant_ID:block) +
#    (1| Participant_ID:category) +
#    (1| Participant_ID:film) ,
#  data = data_acc_glmm,
#  family = binomial(link = "logit")
#)
#library(glmmTMB)

#model_acc_glmm <- glmmTMB(
#  DV_Value ~ category * Group_Enc * block + film +
#    (1 + category + block | Participant_ID) + 
#    (1 | film),
#  family = binomial,
#  data = data_acc_glmm
#)

# too simple, not including random slopes
# Family = binomial (Logistic Regression)
# Optimizer 'bobyqa' is more robust for complex random effects
#model_acc_glmm <- glmer(
#  DV_Value ~ category * Group_Enc * block + 
#    (1 | Participant_ID) + 
#    (1 | film),
#  data = data_acc_glmm,
#  family = binomial,
#  control = glmerControl(optimizer = "bobyqa")
#)

# stable model:
model_acc_glmm_cat <- glmer(
  DV_Value ~ Group_Enc * block * category + film +
    (1 + category | Participant_ID) +
    (1| Participant_ID:film) ,
  data = data_acc_glmm,
  family = binomial(link = "logit")
)

## --- Side-by-side comparison of fixed effects ---
#cmp <- data.frame(
#  term      = names(fixef(model_acc_glmm_block)),
#  block_mod = fixef(model_acc_glmm_block),
#  cat_mod   = fixef(model_acc_glmm_cat)
#)
#cmp$diff <- cmp$block_mod - cmp$cat_mod

#print(cmp, digits = 4)

## pick cat model, since both are similar
model_acc_glmm <- model_acc_glmm_cat


# 3. ANOVA with P-values (Type III Wald Chi-Square)
cat("\n=== ANOVA TABLE (With P-values) ===\n")
# Use Anova() from 'car' package to ensure P-values are printed
Anova(model_acc_glmm, type = "III", test.statistic = "Chisq")

# Model Summary (Coefficients with P-values)
cat("\n=== Model Coefficients ===\n")
print(summary(model_acc_glmm)$coefficients)

# MAIN-EFFECT INTERPRETATION (valid: no significant interactions)

# --- 1. CATEGORY main effect: Face vs Place vs Object (pooled over group & block) ---
cat("\n=== MAIN EFFECT: Category (Face vs Place vs Object) ===\n")
emm_cat <- emmeans(model_acc_glmm, ~ category)            # log-odds (link) scale
print(contrast(emm_cat, method = "pairwise", adjust = "tukey"))
cat("\n--- Predicted probabilities for interpretation ---\n")
print(emm_cat, type = "response")                         # back-transformed accuracy %

# --- 2. BLOCK main effect: Block 1 vs 2 vs 3 (pooled over group & category) ---
cat("\n=== MAIN EFFECT: Block (1 vs 2 vs 3) ===\n")
emm_blk <- emmeans(model_acc_glmm, ~ block)
print(contrast(emm_blk, method = "pairwise", adjust = "tukey"))
cat("\n--- Predicted probabilities for interpretation ---\n")
print(emm_blk, type = "response")

# --- 6. Plot: Predicted Probabilities (0 to 1 scale)
# emmeans automatically back-transforms log-odds to probability when type="response" is used in plotting data
plot_data_acc <- as.data.frame(emmeans(model_acc_glmm, ~ category * Group_Enc * block, type = "response"))

p_acc_glmm <- ggplot(plot_data_acc, aes(x = Group_Enc, y = prob, color = category, group = category)) +
  geom_line(position = position_dodge(0.2), size = 1.2) +
  geom_point(position = position_dodge(0.2), size = 3) +
  geom_errorbar(aes(ymin = asymp.LCL, ymax = asymp.UCL), 
                position = position_dodge(0.2), width = 0.2) +
  facet_wrap(~ block) +
  labs(title = "Predicted Probability of Detection",
       subtitle = "Y-Axis = % Chance of Detecting the Stimulus",
       y = "Percent (0 to 100%)", 
       x = "Group", color = "Category") +
  scale_y_continuous(labels = scales::percent_format()) + # Shows 80% instead of 0.8
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p_acc_glmm)
ggsave(filename = "Encoding_model_acc.png", plot = p_acc_glmm, path = PATH_OUT, dpi = 300)
#### end ####

#### 7. RT ANALYSIS: LMM ON LOG-TRANSFORMED DATA ####
# --- 1. Prepare Data ---
data_rt_glmm <- df_final_analysis %>%
  filter(measure == "RT") %>%
  mutate(
    Group_Enc = case_when(
      timing == "Pro" & job == "Spy" ~ "Spy_Pro",
      timing == "Pro" & job == "Criminal" ~ "Crim_Pro",
      TRUE ~ "Control"
    ),
    Group_Enc = factor(Group_Enc, levels = c("Control", "Spy_Pro", "Crim_Pro")),
    # Log-transform RT to normalize residuals
    # Add a tiny constant if any RTs are exactly 0 (unlikely for RT, but safe practice)
    Log_RT = log(DV_Value*1000) # transform to milliseconds and logtransform
  )

# Check normality of the transformed variable briefly (optional)
# hist(data_rt_glmm$Log_RT, main="Distribution of Log(RT)", breaks=50)

cat("Running RT LMM on Log-Transformed Data...", nrow(data_rt_glmm), "trials\n")

# --- 2. Run Model ---
# Using log(RT) allows us to use standard linear mixed models (lmer)
# which provide reliable P-values via Satterthwaite approximation.
#model_rt_lmm <- lmer( # <- would not converge
#  Log_RT ~ category * Group_Enc * block + film +
#    (1 + category + block| Participant_ID) + 
#    (1 | film),
#  data = data_rt_glmm,
#  REML = TRUE   # FIX #7: tests reported are Satterthwaite F / emmeans, not LR comparisons
#)

#model_rt_lmm <- lmer( # <- was not stable
#  Log_RT ~ category * Group_Enc * block + film +
#    (1| Participant_ID:block) +
#    (1| Participant_ID:category) +
#    (1| Participant_ID:film) ,
#  data = data_rt_glmm,
#  REML = TRUE
#)

model_rt_lmm_cat <- lmer(
  Log_RT ~ category * Group_Enc * block + film +
    (1 + category  | Participant_ID) +
    (1| Participant_ID:film) ,
  data = data_rt_glmm,
  REML = TRUE
)

model_rt_lmm_block <- lmer(
  Log_RT ~ category * Group_Enc * block + film +
    (1 + block | Participant_ID) +
    (1| Participant_ID:film) ,
  data = data_rt_glmm,
  REML = TRUE
)


# --- Side-by-side comparison of fixed effects ---
cmp <- data.frame(
  term      = names(fixef(model_rt_lmm_block)),
  block_mod = fixef(model_rt_lmm_block),
  cat_mod   = fixef(model_rt_lmm_cat)
)
cmp$diff <- cmp$block_mod - cmp$cat_mod

print(cmp, digits = 4)

# pick cat model, since both are similar
model_rt_lmm <- model_rt_lmm_cat

# --- 3. Output Results with P-values ---
cat("\n=== MODEL SUMMARY (Log-Seconds) ===\n")
# Includes t-values and P-values (df approximated by Satterthwaite)
print(summary(model_rt_lmm)$coefficients)

cat("\n=== ANOVA TABLE RT (Type III F-tests) ===\n")
print(anova(model_rt_lmm, type = 3))   # lmerTest::anova -> Type III F-tests, Satterthwaite df

# --- 1. CATEGORY main effect ---
cat("\n=== MAIN EFFECT: Category (Face vs Place vs Object) ===\n")
emm_cat_rt <- emmeans(model_rt_lmm, ~ category)
print(contrast(emm_cat_rt, method = "pairwise", adjust = "tukey"))
cat("\n--- Estimated geometric-mean RTs (ms) ---\n")
print(emm_cat_rt)

# --- 2. BLOCK main effect ---
cat("\n=== MAIN EFFECT: Block (1 vs 2 vs 3) ===\n")
emm_blk_rt <- emmeans(model_rt_lmm, ~ block)
print(contrast(emm_blk_rt, method = "pairwise", adjust = "tukey"))
cat("\n--- Estimated geometric-mean RTs (ms) ---\n")
print(emm_blk_rt)
contrast(emm_blk_rt, method = "poly", degree = 1)   # linear trend across blocks


emm_cat_rt <- emmeans(model_rt_lmm_cat, ~ category)
print(emm_cat_rt)        # no type= / transform= here — already applied
# ============================================================
# RT INTERACTION PROBES (3-way n.s.; probe the three 2-way interactions)
# ============================================================

# --- 1. category : Group_Enc  → Face vs Place vs Object WITHIN each group ---
cat("\n=== category:Group_Enc — category within each Group ===\n")
print(contrast(emmeans(model_rt_lmm, ~ category | Group_Enc),
               method = "pairwise", adjust = "tukey"))
# --- category : Group_Enc, sliced the other way: Group within each category ---
cat("\n=== category:Group_Enc — Group within each Category ===\n")
print(contrast(emmeans(model_rt_lmm, ~ Group_Enc | category),
               method = "pairwise", adjust = "tukey"))

# --- 2. category : block  → category comparisons WITHIN each block ---
cat("\n=== category:block — category within each Block ===\n")
print(contrast(emmeans(model_rt_lmm, ~ category | block),
               method = "pairwise", adjust = "tukey"))

# --- 2. category : block, sliced the other way: Block within each category ---
cat("\n=== category:block — Block within each Category ===\n")
print(contrast(emmeans(model_rt_lmm, ~ block | category),
               method = "pairwise", adjust = "tukey"))

# --- 3. Group_Enc : block  → group comparisons WITHIN each block ---
cat("\n=== Group_Enc:block — Group within each Block ===\n")
print(contrast(emmeans(model_rt_lmm, ~ Group_Enc | block),
               method = "pairwise", adjust = "tukey"))
# --- 3. Group_Enc : block, sliced the other way: Block within each Group ---
cat("\n=== Group_Enc:block — Block within each Group ===\n")
print(contrast(emmeans(model_rt_lmm, ~ block | Group_Enc),
               method = "pairwise", adjust = "tukey"))



# Group differences within each category, separately per block
emm <- emmeans(model_rt_lmm, ~ Group_Enc | category * block)
pairs(emm, adjust = "tukey")


# Or: does the Group × Category pattern differ across blocks?
emm <- emmeans(model_rt_lmm, ~ Group_Enc * category | block)
contrast(emm, interaction = c("consec", "consec"))

#### end ####

# ==================================================================
# PLOT: RT BY CATEGORY (FACETED BY BLOCK)
# Analogous to the Block plot, but showing Category profiles per Block
# ==================================================================

# 1. Prepare Data for this specific plot
# We need subject means and group means grouped by Block AND Category
plot_data_rt_cat <- df_stats_clean %>%
  filter(measure == "RT") %>%
  group_by(block, category, Group_Enc) %>%
  summarise(
    Mean_Val = mean(DV_Value, na.rm = TRUE),
    SE_Val = sd(DV_Value, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

df_subj_means_block_cat <- df_overall %>%
  filter(measure == "RT") %>%
  group_by(Participant_ID, block, category, Group_Enc) %>%
  summarise(Subj_Mean = mean(DV_Value, na.rm = TRUE), .groups = "drop")

# 2. Create the Plot
p_rt_cat_by_block <- ggplot() +
  # A. Individual Subject Means (Jittered background)
  geom_point(data = df_subj_means_block_cat,
             aes(x = category, y = Subj_Mean, color = Group_Enc, group = Group_Enc),
             position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.2),
             alpha = 0.3, size = 1.5) +
  
  # B. Group Means (Line connecting categories)
  geom_line(data = plot_data_rt_cat,
            aes(x = category, y = Mean_Val, color = Group_Enc, group = Group_Enc),
            position = position_dodge(0.2), size = 1.2) +
  
  # C. Group Means (Points)
  geom_point(data = plot_data_rt_cat,
             aes(x = category, y = Mean_Val, color = Group_Enc, group = Group_Enc),
             position = position_dodge(0.2), size = 3, shape = 18) +
  
  # D. Error Bars
  geom_errorbar(data = plot_data_rt_cat,
                aes(x = category, ymin = Mean_Val - SE_Val, ymax = Mean_Val + SE_Val, color = Group_Enc),
                position = position_dodge(0.2), width = 0.1, size = 0.8) +
  
  # E. Facet by Block (One panel per block)
  facet_wrap(~ block, labeller = labeller(block = c("1" = "Block 1", "2" = "Block 2", "3" = "Block 3"))) +
  
  # F. Labels & Theme
  labs(
    title = "Reaction Time by Category (Per Block)",
    subtitle = "Lines connect Face/Place/Object within each Block",
    y = "Reaction Time (seconds)",
    x = "Category",
    color = "Condition"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

print(p_rt_cat_by_block)

# Save the plot
ggsave(filename = "Encoding_RT_by_Category_faceted.png", plot = p_rt_cat_by_block, path = PATH_OUT, dpi = 300)
