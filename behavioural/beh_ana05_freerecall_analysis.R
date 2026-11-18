# ==================================================================
# SHAMEM 1: Ana 05 FREE RECALL ANALYSIS
# ==================================================================

# Notes on exclusion:
# In this script, two participants get excluded. One participant (85) had fallen asleep during
# encoding and therefore didn't complete the task and was excluded from all assessments.
# A second participant (57) was excluded just for the narrative bias scores due to being a statistical
# outlier. The exclusion did not affect the results, and given the assigned condition as retroactive spy,
# whether or not this participant gets excluded based on statistical considerations has no influence on
# the overall results or their interpretation.
# Lastly, for the correlations with eyetracking, only participants with both eyetracking data and narrative
# bias scores were considered, thus excluding the other participants without data.

rm(list = ls())
library(lme4)
library(lmerTest)
library(emmeans)
library(dplyr)
library(ggplot2)
library(tidyr)
library(MASS)
library(sandwich)
library(lmtest)
library(car)        
library(betareg)  
library(ggdist)
library(patchwork)

options(emm_options(pbkrtest.limit = 10000, lmerTest.limit = 10000))

# --- 0. LOAD & PREP DATA ---
# ---- Paths: edit these two lines ----
DATA_ROOT   <- "path/to/data_deposit"   # unpacked data deposit
RESULTS_DIR <- "path/to/results"        # e.g. the repository's results/ folder

PATH_IN     <- file.path(DATA_ROOT, "behavioural_results")   # shamem_retrieval_freerecall.csv
PATH_OUT    <- file.path(RESULTS_DIR, "behavioural")
# Eye-tracking bias scores (ET_bias_scores_participant.csv and
# ET_bias_scores_participant_block.csv), written by eyetracking/analysis/encoding_bias.py.
# A copy of the files used in the paper is also in the data deposit.
PATH_IN_eye <- file.path(RESULTS_DIR, "eyetracking_results")
dir.create(PATH_OUT, recursive = TRUE, showWarnings = FALSE)

setwd(PATH_IN)

df_raw <- read.csv("shamem_retrieval_freerecall.csv")

df <- df_raw %>%
  rename(
    Participant_ID = "id",
    job            = "job",
    timing         = "timing",
    Category       = "category",
    Correctness    = "correct",
    DV_Value       = "output", 
    MeasureType    = "measure"
  ) %>%
  mutate(
    Participant_ID = factor(trimws(as.character(Participant_ID))),
    job = factor(job, levels = c(1, 2, 3), labels = c("Spy", "Criminal", "Neutral")),
    timing = factor(timing, levels = c(1, 2, 3), labels = c("Pro", "Retro", "Neutral")),
    Group_Ret = case_when(
      job == "Neutral" ~ "Control",
      timing == "Pro" & job == "Spy" ~ "Spy_Pro",
      timing == "Pro" & job == "Criminal" ~ "Crim_Pro",
      timing == "Retro" & job == "Spy" ~ "Spy_Retro",
      timing == "Retro" & job == "Criminal" ~ "Crim_Retro"
    ),
    Group_Ret = factor(Group_Ret, levels = c("Control", "Spy_Pro", "Crim_Pro", "Spy_Retro", "Crim_Retro")),
    Category = tolower(as.character(Category)),
    Correctness = tolower(as.character(Correctness)),
    MeasureType = tolower(as.character(MeasureType))
  )

cat("Data Loaded. Unique Participants:", length(unique(df$Participant_ID)), "\n")

# Exclusions based on who fell asleep during encoding
excluded_subs <- c(85)
df <- df %>% filter(!Participant_ID %in% excluded_subs)

setwd(PATH_OUT)


# ==================================================================
# PART 1: COUNT DATA — NEGATIVE BINOMIAL GLM
# ==================================================================
cat("\n================================================================\n")
cat("=== Part 1: COUNT SEMANTIC CATEGORY DIFFERENCES ===\n")
cat("================================================================\n")

run_count_analysis <- function(df_input, timing_label, contrast_groups, filename_prefix) {
  # contrast_groups: c("Spy_Pro", "Crim_Pro") etc.
  
  results_list <- list()
  contrast_results <- list()
  
  if(nrow(df_input) == 0) {
    cat("No data for", timing_label, "\n")
    return(invisible(NULL))
  }
  
  for (cat_name in unique(df_input$Category)) {
    cat("\n--- Category:", cat_name, "---\n")
    d_sub <- df_input %>% filter(Category == cat_name)
    
    # Aggregate to one row per participant
    if(nrow(d_sub) != length(unique(d_sub$Participant_ID))) {
      d_sub <- d_sub %>% group_by(Participant_ID, Group_Ret) %>% 
        summarise(DV_Value = sum(DV_Value), .groups = "drop")
    }
    
    # Drop unused factor levels
    d_sub <- d_sub %>% mutate(Group_Ret = droplevels(Group_Ret))
    
    if(sd(d_sub$DV_Value) == 0) {
      cat("Skipping: Zero variance.\n")
      next
    }
    
    # Check if we have enough groups
    n_groups <- length(unique(d_sub$Group_Ret))
    if(n_groups < 2) {
      cat("Skipping: Only one group present.\n")
      next
    }
    
    mod <- NULL
    model_type <- "Negative Binomial"
    
    # Fit Negative Binomial
    tryCatch({
      suppressWarnings({
        mod <- glm.nb(DV_Value ~ Group_Ret, data = d_sub)
      })
      cat("-> NegBin fitted. Theta:", round(mod$theta, 3), "\n")
      
      # NEW: Report overdispersion diagnostic
      dispersion <- sum(residuals(mod, type = "pearson")^2) / mod$df.residual
      cat("   Dispersion ratio:", round(dispersion, 3), "\n")
      
    }, error = function(e) {
      cat("-> NegBin failed. Fallback to Poisson.\n")
      suppressWarnings({
        mod <<- glm(DV_Value ~ Group_Ret, data = d_sub, family = poisson)
      })
      model_type <<- "Poisson"
    })
    
    if(is.null(mod)) next
    
    # Omnibus test via Likelihood Ratio Test
    model_stats <- tryCatch({
      null_mod <- update(mod, . ~ 1)
      
      # Manual LRT: 2 * (logLik(full) - logLik(null))
      ll_full <- logLik(mod)
      ll_null <- logLik(null_mod)
      
      chi_sq_val <- as.numeric(2 * (ll_full - ll_null))
      df_val <- attr(ll_full, "df") - attr(ll_null, "df")
      p_val <- pchisq(chi_sq_val, df = df_val, lower.tail = FALSE)
      
      list(chi2 = chi_sq_val, df = as.integer(df_val), p = p_val, test_type = "LRT")
      
    }, error = function(e) {
      cat("   LRT failed:", e$message, "\n")
      # Fallback: car::Anova with Wald (correctly handles covariance)
      tryCatch({
        aov_res <- car::Anova(mod, type = "III", test.statistic = "Wald")
        row_idx <- which(rownames(aov_res) == "Group_Ret")
        if(length(row_idx) > 0) {
          list(chi2 = aov_res[row_idx, "Chisq"], 
               df = as.integer(aov_res[row_idx, "Df"]), 
               p = aov_res[row_idx, "Pr(>Chisq)"],
               test_type = "Wald (car::Anova)")
        } else {
          list(chi2 = NA_real_, df = NA_integer_, p = NA_real_, test_type = NA_character_)
        }
      }, error = function(e2) {
        list(chi2 = NA_real_, df = NA_integer_, p = NA_real_, test_type = NA_character_)
      })
    })
    
    # Guard against NULL or missing elements
    if(!is.null(model_stats) && !is.na(model_stats$chi2)) {
      cat(sprintf("   [OMNIBUS] %s χ²(%d) = %.2f, p = %.4f\n", 
                  model_stats$test_type, model_stats$df, model_stats$chi2, model_stats$p))
    }
    
    # ------------------------------------------------------------------
    # Planned Contrast via emmeans
    # ------------------------------------------------------------------
    tryCatch({
      emm <- emmeans(mod, ~ Group_Ret, type = "response")
      lvl <- levels(d_sub$Group_Ret)
      
      # Build contrast vector
      contrast_vec <- setNames(rep(0, length(lvl)), lvl)
      g1 <- contrast_groups[1]
      g2 <- contrast_groups[2]
      
      if(g1 %in% lvl) contrast_vec[g1] <- 1
      if(g2 %in% lvl) contrast_vec[g2] <- -1
      
      if(sum(contrast_vec != 0) >= 2) {
        contr <- contrast(emm, method = list("Spy_vs_Criminal" = contrast_vec))
        contr_sum <- summary(contr, infer = TRUE)
        
        # GLM contrasts use asymptotic z -> column is z.ratio, not t.ratio
        test_stat <- if(!is.null(contr_sum$t.ratio)) contr_sum$t.ratio else contr_sum$z.ratio
        stat_label <- if(!is.null(contr_sum$t.ratio)) "t" else "z"
        
        ci <- as.data.frame(confint(contr))
        ci_low  <- as.numeric(ci[grep("LCL|lower", names(ci), value = TRUE)][1])
        ci_high <- as.numeric(ci[grep("UCL|upper", names(ci), value = TRUE)][1])
        
        emm_df <- as.data.frame(emm)
        mean_g1 <- emm_df[emm_df$Group_Ret == g1, "response"] %>% as.numeric()
        mean_g2 <- emm_df[emm_df$Group_Ret == g2, "response"] %>% as.numeric()
        
        contrast_results[[cat_name]] <- tibble(
          Category = cat_name,
          Model_Type = model_type,
          Theta = ifelse(model_type == "Negative Binomial", round(mod$theta, 3), NA_real_),
          Omnibus_Test = model_stats$test_type,
          Omnibus_Chi2 = round(model_stats$chi2, 2),
          Omnibus_DF = model_stats$df,
          Omnibus_P = round(model_stats$p, 3),
          Contrast_Estimate = round(contr_sum$ratio, 2),
          Contrast_SE = round(contr_sum$SE, 2),
          Contrast_Stat = round(test_stat, 2),
          Contrast_StatType = stat_label,        # records whether it's z or t
          Contrast_P = round(contr_sum$p.value, 3),
          Contrast_CI_Low = round(ci_low, 2),
          Contrast_CI_High = round(ci_high, 2),
          Mean_G1 = round(mean_g1, 2),
          Mean_G2 = round(mean_g2, 2)
        )
        
        cat(sprintf("   [CONTRAST] %s vs %s: Diff = %.2f counts, %s = %.2f, p = %.4f\n", 
                    g1, g2, contr_sum$estimate, stat_label, test_stat, contr_sum$p.value))
      }
      
    }, error = function(e) {
      cat("   Contrast failed:", e$message, "\n")
    })
  }
  
  # Save results
  if(length(contrast_results) > 0) {
    final_stats <- bind_rows(contrast_results)
    write.csv(final_stats, file.path(PATH_OUT, paste0(filename_prefix, ".csv")), row.names = FALSE)
    cat("\nResults saved:", paste0(filename_prefix, ".csv"), "\n")
    return(final_stats)
  }
}

#### --- PRO --- ####
df_part_pro <- df %>%
  filter(Correctness == "total", !timing == "Retro", MeasureType == "count")

results_pro <- run_count_analysis(
  df_input = df_part_pro,
  timing_label = "PRO",
  contrast_groups = c("Spy_Pro", "Crim_Pro"),
  filename_prefix = "semantic_category_pro_count_results"
)

### Plotting
# 1. order for the Radial Plot
plot_order <- c("speech", "spatial", "insecurity", "other", "unrelated", "person", "object")

# 2. PLOTTING: Closed Radial Plot

df_plot_data <- df_part_pro %>%
  filter(Category %in% plot_order)

if(nrow(df_plot_data) > 0) {
  df_plot_agg <- df_plot_data %>%
    group_by(Participant_ID, Group_Ret, Category) %>%
    summarise(DV_Value = sum(DV_Value), .groups = "drop")
  
  summary_data <- df_plot_agg %>%
    group_by(Group_Ret, Category) %>%
    summarise(
      Median = median(DV_Value, na.rm = TRUE),
      Mean = mean(DV_Value, na.rm = TRUE),
      SD = sd(DV_Value, na.rm = TRUE),
      N = n(),
      SEM = SD / sqrt(N),
      .groups = 'drop'
    ) %>%
    mutate(
      Category = factor(Category, levels = plot_order),
      Group_Ret = factor(Group_Ret, levels = c("Control", "Spy_Pro", "Crim_Pro"))
    )
  
  max_y <- max(summary_data$Mean + summary_data$SEM, na.rm = TRUE)
  zoom_limit <- max_y * 1.1 
  
  group_colors <- c("Spy_Pro" = "#0072B2", "Crim_Pro" = "#D55E00", "Control" = "#A6A6A6")
  present_groups <- unique(summary_data$Group_Ret)
  group_colors <- group_colors[names(group_colors) %in% present_groups]
  
  radial_plot <- ggplot(summary_data, aes(x = Category, y = Mean, group = Group_Ret, color = Group_Ret, fill = Group_Ret)) +
    geom_ribbon(aes(ymin = Mean - SEM, ymax = Mean + SEM), alpha = 0.15, color = NA) +
    geom_line(aes(y = Mean), size = 0.8, alpha = 1.0) +
    geom_point(aes(y = Mean), size = 1.5, alpha = 1.0) +
    scale_color_manual(values = group_colors, name = "Group") +
    scale_fill_manual(values = group_colors, name = "Group") +
    ylim(0, zoom_limit) +
    coord_polar(start = -pi/2, direction = -1) +
    theme_minimal(base_size = 12) +
    theme(
      text = element_text(family = "Arial", size = 22),
      axis.text.y = element_text(color = "grey40"),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    ) +
    labs(
      title = "Semantic Recall Counts by Category",
      subtitle = "Solid Line = Mean | Shaded Area = Mean ± SEM",
      y = "Count",
      x = ""
    )
  
  print(radial_plot)
  ggsave(file.path(PATH_OUT, "radial_plot_pro_count.svg"), radial_plot, width = 8, height = 8, dpi = 300)
  cat("Plot saved.\n")
}
#### end ####



#### --- RETRO --- ####
df_part_retro <- df %>%
  filter(Correctness == "total", !timing == "Pro", MeasureType == "count")

results_retro <- run_count_analysis(
  df_input = df_part_retro,
  timing_label = "RETRO",
  contrast_groups = c("Spy_Retro", "Crim_Retro"),
  filename_prefix = "semantic_category_retro_count_results"
)


### Plotting
# 1. order for the Radial Plot
plot_order <- c("speech", "spatial", "insecurity", "other", "unrelated", "person", "object")

# 2. PLOTTING: Closed Radial Plot

df_plot_data <- df_part_retro %>%
  filter(Category %in% plot_order)

if(nrow(df_plot_data) > 0) {
  df_plot_agg <- df_plot_data %>%
    group_by(Participant_ID, Group_Ret, Category) %>%
    summarise(DV_Value = sum(DV_Value), .groups = "drop")
  
  summary_data <- df_plot_agg %>%
    group_by(Group_Ret, Category) %>%
    summarise(
      Median = median(DV_Value, na.rm = TRUE),
      Mean = mean(DV_Value, na.rm = TRUE),
      SD = sd(DV_Value, na.rm = TRUE),
      N = n(),
      SEM = SD / sqrt(N),
      .groups = 'drop'
    ) %>%
    mutate(
      Category = factor(Category, levels = plot_order),
      Group_Ret = factor(Group_Ret, levels = c("Control", "Spy_Retro", "Crim_Retro"))
    )
  
  max_y <- max(summary_data$Mean + summary_data$SEM, na.rm = TRUE)
  zoom_limit <- max_y * 1.1 
  
  group_colors <- c("Spy_Retro" = "#56B4E9", "Crim_Retro" = "#E69F00", "Control" = "#A6A6A6")
  present_groups <- unique(summary_data$Group_Ret)
  group_colors <- group_colors[names(group_colors) %in% present_groups]
  
  radial_plot <- ggplot(summary_data, aes(x = Category, y = Mean, group = Group_Ret, color = Group_Ret, fill = Group_Ret)) +
    geom_ribbon(aes(ymin = Mean - SEM, ymax = Mean + SEM), alpha = 0.15, color = NA) +
    geom_line(aes(y = Mean), size = 0.8, alpha = 1.0) +
    geom_point(aes(y = Mean), size = 1.5, alpha = 1.0) +
    scale_color_manual(values = group_colors, name = "Group") +
    scale_fill_manual(values = group_colors, name = "Group") +
    ylim(0, zoom_limit) +
    coord_polar(start = -pi/2, direction = -1) +
    theme_minimal(base_size = 12) +
    theme(
      text = element_text(family = "Arial", size = 22),
      axis.text.y = element_text(color = "grey40"),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    ) +
    labs(
      title = "Semantic Recall Counts by Category",
      subtitle = "Solid Line = Mean | Shaded Area = Mean ± SEM",
      y = "Count",
      x = ""
    )
  
  print(radial_plot)
  ggsave(file.path(PATH_OUT, "radial_plot_retro_count.svg"), radial_plot, width = 8, height = 8, dpi = 300)
  cat("Plot saved.\n")
}

#### end ####

# ==================================================================
# PART 2: PERCENTAGE DATA — BETA REGRESSION / QUASI-BINOMIAL
# ==================================================================
cat("\n================================================================\n")
cat("=== Part 2: PERCENTAGE SEMANTIC CATEGORY DIFFERENCES ===\n")
cat("================================================================\n")

run_percentage_analysis <- function(df_input, timing_label, contrast_groups, filename_prefix) {
  
  results_list <- list()
  contrast_results <- list()
  
  if(nrow(df_input) == 0) {
    cat("No data for", timing_label, "\n")
    return(invisible(NULL))
  }
  
  for (cat_name in unique(df_input$Category)) {
    cat("\n--- Category:", cat_name, "---\n")
    d_sub <- df_input %>% filter(Category == cat_name)
    
    # Aggregate to one row per participant
    if(nrow(d_sub) != length(unique(d_sub$Participant_ID))) {
      d_sub <- d_sub %>% group_by(Participant_ID, Group_Ret) %>% 
        summarise(DV_Value = mean(DV_Value), .groups = "drop")
    }
    
    d_sub <- d_sub %>% mutate(Group_Ret = droplevels(Group_Ret))
    
    # Convert to proportion
    d_sub <- d_sub %>% mutate(Prop = DV_Value / 100)
    
    n_at_zero <- sum(d_sub$Prop == 0, na.rm = TRUE)
    n_at_one <- sum(d_sub$Prop == 1, na.rm = TRUE)
    
    if(n_at_zero > 0 || n_at_one > 0) {
      cat(sprintf("   Boundary values: %d at 0%%, %d at 100%%\n", n_at_zero, n_at_one))
      # Use adjustment based on sample size (Cribari-Neto & Zeileis, 2010 recommend (0.5/n))
      adj <- 0.5 / nrow(d_sub)
      d_sub <- d_sub %>% 
        mutate(Prop_Adj = pmin(pmax(Prop, adj), 1 - adj))
    } else {
      d_sub <- d_sub %>% mutate(Prop_Adj = Prop)
    }
    
    if(sd(d_sub$Prop_Adj) == 0) {
      cat("Skipping: Zero variance.\n")
      next
    }
    
    # ------------------------------------------------------------------
    # Use Beta Regression as primary model (good for proportions)
    # Falls back to quasi-binomial if beta regression fails
    # ------------------------------------------------------------------
    mod <- NULL
    model_type <- NA
    
    # Try Beta Regression first
    tryCatch({
      suppressWarnings({
        mod <- betareg(Prop_Adj ~ Group_Ret, data = d_sub)
      })
      model_type <- "Beta Regression (logit link)"
      cat("-> Beta regression fitted.\n")
    }, error = function(e) {
      cat("-> Beta regression failed:", e$message, "\n")
      # Fallback to quasi-binomial
      tryCatch({
        suppressWarnings({
          mod <<- glm(Prop_Adj ~ Group_Ret, data = d_sub, family = quasibinomial)
        })
        model_type <<- "Quasi-Binomial GLM"
        cat("-> Quasi-Binomial GLM fitted. Dispersion:", round(summary(mod)$dispersion, 3), "\n")
      }, error = function(e2) {
        cat("-> Quasi-binomial also failed:", e2$message, "\n")
      })
    })
    
    if(is.null(mod)) next
    
    # ------------------------------------------------------------------
    # Omnibus test appropriate to model type
    # ------------------------------------------------------------------
    model_stat <- NA
    model_df1 <- NA
    model_df2 <- NA
    model_p <- NA
    test_type <- NA
    
    tryCatch({
      if(grepl("Beta", model_type)) {
        # For betareg: use LR test against null model
        null_mod <- update(mod, . ~ 1 | 1)
        lr_stat <- 2 * (logLik(mod) - logLik(null_mod))
        df_diff <- attr(logLik(mod), "df") - attr(logLik(null_mod), "df")
        model_stat <- as.numeric(lr_stat)
        model_df1 <- df_diff
        model_p <- pchisq(lr_stat, df = df_diff, lower.tail = FALSE)
        test_type <- "LR test (χ²)"
      } else {
        # For quasi-binomial: F-test is correct
        anova_res <- anova(mod, test = "F")
        row_idx <- which(rownames(anova_res) == "Group_Ret")
        if(length(row_idx) > 0) {
          model_stat <- anova_res[row_idx, "F"]
          model_df1 <- anova_res[row_idx, "Df"]
          model_df2 <- mod$df.residual
          model_p <- anova_res[row_idx, "Pr(>F)"]
          test_type <- "F-test"
        }
      }
      
      if(!is.na(model_p)) {
        cat(sprintf("   [OMNIBUS] %s", test_type))
        if(test_type == "F-test") {
          cat(sprintf(" F(%d, %d) = %.2f, p = %.4f\n", model_df1, model_df2, model_stat, model_p))
        } else {
          cat(sprintf(" χ²(%d) = %.2f, p = %.4f\n", model_df1, model_stat, model_p))
        }
      }
    }, error = function(e) {
      cat("   Omnibus test failed:", e$message, "\n")
    })
    
    # ------------------------------------------------------------------
    # Contrast via emmeans on response scale (not manual logit math)
    # (proper percentage-point differences)
    # ------------------------------------------------------------------
    tryCatch({
      emm <- emmeans(mod, ~ Group_Ret, type = "response")
      lvl <- levels(d_sub$Group_Ret)
      contrast_vec <- setNames(rep(0, length(lvl)), lvl)
      g1 <- contrast_groups[1]
      g2 <- contrast_groups[2]
      
      if(g1 %in% lvl) contrast_vec[g1] <- 1
      if(g2 %in% lvl) contrast_vec[g2] <- -1
      
      if(sum(contrast_vec != 0) >= 2) {
        contr <- contrast(emm, method = list("Spy_vs_Criminal" = contrast_vec))
        contr_sum <- summary(contr, infer = TRUE)
        
        # GLM/beta contrasts use asymptotic z -> z.ratio, not t.ratio
        test_stat  <- if(!is.null(contr_sum$t.ratio)) contr_sum$t.ratio else contr_sum$z.ratio
        stat_label <- if(!is.null(contr_sum$t.ratio)) "t" else "z"
        
        ci <- as.data.frame(confint(contr))
        ci_low  <- as.numeric(ci[grep("LCL|lower", names(ci), value = TRUE)][1])
        ci_high <- as.numeric(ci[grep("UCL|upper", names(ci), value = TRUE)][1])
        
        # Group means for reporting
        emm_df <- as.data.frame(emm)
        mean_g1 <- emm_df[emm_df$Group_Ret == g1, "emmean"] %>% as.numeric()
        mean_g2 <- emm_df[emm_df$Group_Ret == g2, "emmean"] %>% as.numeric()
        
        contrast_results[[cat_name]] <- tibble(
          Category = cat_name,
          Model_Type = model_type,
          Omnibus_Test = test_type,
          Omnibus_Stat = round(model_stat, 2),
          Omnibus_DF1 = as.integer(model_df1),
          Omnibus_DF2 = ifelse(is.na(model_df2), NA_integer_, as.integer(model_df2)),
          Omnibus_P = round(model_p, 3),
          Contrast_Estimate_pp = round(contr_sum$estimate * 100, 2),
          Contrast_SE_pp = round(contr_sum$SE * 100, 2),
          Contrast_Stat = round(test_stat, 2),
          Contrast_StatType = stat_label,        # records z vs t
          Contrast_P = round(contr_sum$p.value, 3),
          Contrast_CI_Low_pp = round(ci_low * 100, 2),
          Contrast_CI_High_pp = round(ci_high * 100, 2),
          Mean_G1_pct = round(mean_g1 * 100, 2),
          Mean_G2_pct = round(mean_g2 * 100, 2)
        )
        
        cat(sprintf("   [CONTRAST] %s vs %s: Diff = %.2f pp, t = %.2f, p = %.4f\n", 
                    g1, g2, contr_sum$estimate * 100, contr_sum$t.ratio, contr_sum$p.value))
      }
    }, error = function(e) {
      cat("   Contrast failed:", e$message, "\n")
    })
  }
  
  if(length(contrast_results) > 0) {
    final_stats <- bind_rows(contrast_results)
    write.csv(final_stats, file.path(PATH_OUT, paste0(filename_prefix, ".csv")), row.names = FALSE)
    cat("\nResults saved:", paste0(filename_prefix, ".csv"), "\n")
    return(final_stats)
  }
}

#### --- PRO --- ####
df_pct_pro <- df %>%
  filter(Correctness == "total", !timing == "Retro", MeasureType == "percent")

results_pct_pro <- run_percentage_analysis(
  df_input = df_pct_pro,
  timing_label = "PRO",
  contrast_groups = c("Spy_Pro", "Crim_Pro"),
  filename_prefix = "semantic_category_pro_percentage_results"
)

### Plotting
# 1. Define the specific order for the Radial Plot
plot_order <- c("speech", "spatial", "insecurity", "other", "unrelated", "person", "object")

# 2. PLOTTING: Radial Plot for Percentages

df_plot_pct <- df_pct_pro %>%
  filter(Category %in% plot_order)

if(nrow(df_plot_pct) > 0) {
  df_plot_agg <- df_plot_pct %>%
    group_by(Participant_ID, Group_Ret, Category) %>%
    summarise(DV_Value = mean(DV_Value), .groups = "drop")
  
  summary_data <- df_plot_agg %>%
    group_by(Group_Ret, Category) %>%
    summarise(
      Median = median(DV_Value, na.rm = TRUE),
      Mean = mean(DV_Value, na.rm = TRUE),
      SD = sd(DV_Value, na.rm = TRUE),
      N = n(),
      SEM = SD / sqrt(N),
      .groups = 'drop'
    ) %>%
    mutate(
      Category = factor(Category, levels = plot_order),
      Group_Ret = factor(Group_Ret, levels = c("Control", "Spy_Pro", "Crim_Pro"))
    )
  
  max_y <- max(summary_data$Mean + summary_data$SEM, na.rm = TRUE)
  zoom_limit <- max_y * 1.1 
  
  group_colors <- c("Spy_Pro" = "#0072B2", "Crim_Pro" = "#D55E00", "Control" = "#A6A6A6")
  present_groups <- unique(summary_data$Group_Ret)
  group_colors <- group_colors[names(group_colors) %in% present_groups]
  
  radial_plot_pct <- ggplot(summary_data, aes(x = Category, y = Mean, group = Group_Ret, color = Group_Ret, fill = Group_Ret)) +
    geom_ribbon(aes(ymin = Mean - SEM, ymax = Mean + SEM), alpha = 0.15, color = NA) +
    geom_line(aes(y = Mean), size = 0.8, alpha = 1.0) +
    geom_point(aes(y = Mean), size = 1.5, alpha = 1.0) +
    scale_color_manual(values = group_colors, name = "Group") +
    scale_fill_manual(values = group_colors, name = "Group") +
    ylim(0, zoom_limit) +
    coord_polar(start = -pi/2, direction = -1) +
    theme_minimal(base_size = 12) +
    theme(
      text = element_text(family = "Arial", size = 22),
      axis.text.y = element_text(color = "grey40"),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    ) +
    labs(
      title = "Semantic Recall Percentages by Category",
      subtitle = "Solid Line = Mean | Shaded Area = Mean ± SEM | Y-axis = %",
      y = "% of speech",
      x = ""
    )
  
  print(radial_plot_pct)
  ggsave(file.path(PATH_OUT, "radial_plot_pro_percentage.svg"), radial_plot_pct, width = 8, height = 8, dpi = 300)
  cat("Percentage plot saved.\n")
}

#### end ####

#### --- RETRO --- ####
df_pct_retro <- df %>%
  filter(Correctness == "total", !timing == "Pro", MeasureType == "percent")

results_pct_retro <- run_percentage_analysis(
  df_input = df_pct_retro,
  timing_label = "RETRO",
  contrast_groups = c("Spy_Retro", "Crim_Retro"),
  filename_prefix = "semantic_category_retro_percentage_results"
)

### Plotting
# 1. Define the specific order for the Radial Plot
plot_order <- c("speech", "spatial", "insecurity", "other", "unrelated", "person", "object")

# 2. PLOTTING: Radial Plot for Percentages
df_plot_pct <- df_pct_retro %>%
  filter(Category %in% plot_order)

if(nrow(df_plot_pct) > 0) {
  df_plot_agg <- df_plot_pct %>%
    group_by(Participant_ID, Group_Ret, Category) %>%
    summarise(DV_Value = mean(DV_Value), .groups = "drop")
  
  summary_data <- df_plot_agg %>%
    group_by(Group_Ret, Category) %>%
    summarise(
      Median = median(DV_Value, na.rm = TRUE),
      Mean = mean(DV_Value, na.rm = TRUE),
      SD = sd(DV_Value, na.rm = TRUE),
      N = n(),
      SEM = SD / sqrt(N),
      .groups = 'drop'
    ) %>%
    mutate(
      Category = factor(Category, levels = plot_order),
      Group_Ret = factor(Group_Ret, levels = c("Control", "Spy_Retro", "Crim_Retro"))
    )
  
  max_y <- max(summary_data$Mean + summary_data$SEM, na.rm = TRUE)
  zoom_limit <- max_y * 1.1 
  
  group_colors <- c("Spy_Retro" = "#56B4E9", "Crim_Retro" = "#E69F00", "Control" = "#A6A6A6")
  present_groups <- unique(summary_data$Group_Ret)
  group_colors <- group_colors[names(group_colors) %in% present_groups]
  
  radial_plot_pct <- ggplot(summary_data, aes(x = Category, y = Mean, group = Group_Ret, color = Group_Ret, fill = Group_Ret)) +
    geom_ribbon(aes(ymin = Mean - SEM, ymax = Mean + SEM), alpha = 0.15, color = NA) +
    geom_line(aes(y = Mean), size = 0.8, alpha = 1.0) +
    geom_point(aes(y = Mean), size = 1.5, alpha = 1.0) +
    scale_color_manual(values = group_colors, name = "Group") +
    scale_fill_manual(values = group_colors, name = "Group") +
    ylim(0, zoom_limit) +
    coord_polar(start = -pi/2, direction = -1) +
    theme_minimal(base_size = 12) +
    theme(
      text = element_text(family = "Arial", size = 22),
      axis.text.y = element_text(color = "black"),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    ) +
    labs(
      title = "Semantic Recall Percentages by Category",
      subtitle = "Solid Line = Mean | Shaded Area = Mean ± SEM | Y-axis = %",
      y = "% of speech",
      x = ""
    )
  
  print(radial_plot_pct)
  ggsave(file.path(PATH_OUT, "radial_plot_retro_percentage.svg"), radial_plot_pct, width = 8, height = 8, dpi = 300)
  cat("Percentage plot saved.\n")
}
#### end ####


#### Part 3: NARRATIVE BIAS SCORE (Quasi-binomial GLMs) ####
cat("\n================================================================\n")
cat("=== PART 3: NARRATIVE BIAS SCORES ===\n")
cat("================================================================\n")

df_bias_raw <- df %>%
  dplyr::filter(Category %in% c("person", "spatial"),
                MeasureType == "percent",
                !is.na(DV_Value))

bias_results <- list()
plot_data_list <- list()
bias_updates_list <- list()
bias_summary_list <- list()

# helper to turn a p-value into a significance star for plotting
star_label <- function(p) {
  if (is.na(p)) return("")
  ifelse(p < .001, "***", ifelse(p < .01, "**", ifelse(p < .05, "*", ifelse(p < .10, "+", ""))))
}


# Effect size for the omnibus group effect: partial eta-squared
partial_eta2 <- function(model) {
  aov_res <- anova(model)
  ss <- aov_res[["Sum Sq"]]; names(ss) <- rownames(aov_res)
  ss["Group_Ret"] / (ss["Group_Ret"] + ss["Residuals"])
}

# repeat calculations for all statements, only factually accurate, and for false ones (so 3x in total)
for (corr_level in c("total", "right", "wrong")) {
  cat("\n### CORRECTNESS LEVEL:", toupper(corr_level), "###\n")
  
  d_sub <- df_bias_raw %>% dplyr::filter(Correctness == corr_level)
  if(nrow(d_sub) == 0) { cat("No data. Skipping.\n"); next }
  
  d_wide <- d_sub %>%
    dplyr::select(Participant_ID, Group_Ret, Category, DV_Value) %>%
    tidyr::pivot_wider(names_from = Category, values_from = DV_Value, values_fn = mean) %>%
    dplyr::filter(!is.na(person) & !is.na(spatial)) %>%
    dplyr::mutate(Bias_Score = person / spatial)
  
  # exclude statistical outlier participant bias score of > 10 (group: retroactive spy)
  # rest in retroactive spy group: Spy_Retro (n= 19), M = 3.68, SD = 0.879, SE = 0.202
  d_wide <- d_wide %>% filter(!Participant_ID == 57)
  
  df_bias_update <- d_wide %>%
    dplyr::select(Participant_ID, Bias_Score) %>%
    dplyr::mutate(Correctness = corr_level)
  bias_updates_list[[corr_level]] <- df_bias_update
  cat("Calculated bias for", nrow(df_bias_update), "participants at level", corr_level, "\n")
  
  if(nrow(d_wide) < 5) { cat("Insufficient participants. Skipping.\n"); next }
  
  # compute Pro-phase model BEFORE plotting, capture the two p-values
  p_spy_crim_pro <- NA_real_   # for the Spy<->Criminal bracket
  p_crim_ctrl_pro <- NA_real_  # for the Criminal<->Control bracket
  d_pro_pre <- d_wide %>% dplyr::filter(Group_Ret %in% c("Spy_Pro", "Crim_Pro", "Control"))
  d_pro_pre$Group_Ret <- factor(d_pro_pre$Group_Ret)
  if(length(unique(d_pro_pre$Group_Ret)) >= 2) {
    mod_pro_pre <- lm(Bias_Score ~ Group_Ret, data = d_pro_pre)
    emm_pre <- emmeans(mod_pro_pre, ~ Group_Ret)
    # Spy vs Criminal: HC3-robust (matches planned contrast)
    vec_sc <- setNames(rep(0, length(levels(d_pro_pre$Group_Ret))), levels(d_pro_pre$Group_Ret))
    vec_sc["Spy_Pro"] <- 1; vec_sc["Crim_Pro"] <- -1
    p_spy_crim_pro <- as.numeric(summary(contrast(emm_pre, method = list("sc" = vec_sc),
                                                  vcov = vcovHC(mod_pro_pre, type = "HC3"), adjust = "holm"))$p.value)
    # Criminal vs Control: Tukey pairwise (matches post-hoc)
    pw <- as.data.frame(contrast(emm_pre, method = "pairwise", adjust = "tukey"))
    row_i <- grep("Crim_Pro.*Control|Control.*Crim_Pro", pw$contrast)
    if(length(row_i) >= 1) p_crim_ctrl_pro <- pw$p.value[row_i[1]]
  }
  
  # ---------------------------------------------------------
  # PLOTTING: BIAS SCORES
  # ---------------------------------------------------------
  d_plot <- d_wide %>%
    mutate(
      Original_Code = case_when(
        grepl("Spy_Pro", Group_Ret) ~ "Spy_Pro",
        grepl("Crim_Pro", Group_Ret) ~ "Crim_Pro",
        grepl("Control", Group_Ret) ~ "Control",
        grepl("Spy_Retro", Group_Ret) ~ "Spy_Retro",
        grepl("Crim_Retro", Group_Ret) ~ "Crim_Retro",
        TRUE ~ as.character(Group_Ret)),
      Group_Display = factor(Group_Ret,
                             levels = c("Spy_Pro", "Crim_Pro", "Control", "Spy_Retro", "Crim_Retro")),
      x_num = as.numeric(Group_Display),
      x_dot_jitter = x_num + 0.25 + runif(n(), -0.1, 0.1))
  
  my_palette <- c(
    "Spy_Pro" = "#0072B2", "Crim_Pro" = "#D55E00", "Control" = "#A6A6A6",
    "Spy_Retro" = "#56B4E9", "Crim_Retro" = "#E69F00")
  
  means_df <- d_plot %>%
    filter(Original_Code %in% c("Spy_Pro", "Crim_Pro", "Spy_Retro", "Crim_Retro")) %>%
    group_by(Group_Display, Original_Code) %>%
    summarise(Mean_Bias = median(Bias_Score, na.rm = TRUE), .groups = "drop") %>%
    mutate(Line_Group = case_when(
      Original_Code %in% c("Spy_Pro", "Crim_Pro") ~ "Pro_Pair",
      Original_Code %in% c("Spy_Retro", "Crim_Retro") ~ "Retro_Pair"))
  
  simple_labels <- c("Spy_Pro" = "Spy", "Crim_Pro" = "Criminal", "Control" = "Control",
                     "Spy_Retro" = "Spy", "Crim_Retro" = "Criminal")
  
  y_max_val <- max(d_plot$Bias_Score, na.rm = TRUE)
  y_line_1 <- 9; y_text_1 <- y_line_1 + 0.12
  y_line_2 <- 8.5; y_text_2 <- y_line_2 + 0.3
  my_font_family <- "serif"
  
  p_bias <- ggplot(d_plot, aes(x = Group_Display, y = Bias_Score)) +
    geom_violin(aes(fill = Original_Code), trim = FALSE, adjust = 1.2, color = NA, alpha = 0.8,
                quantile.colour = "black", quantile.linewidth = 0.5, quantile.linetype = "solid",
                quantiles = c(0.25, 0.75)) +
    geom_rect(aes(xmin = x_num, xmax = x_num + 0.5, ymin = -Inf, ymax = Inf),
              fill = "white", alpha = 1, color = NA, inherit.aes = FALSE) +
    geom_point(aes(x = x_dot_jitter, color = Original_Code), alpha = 0.7, size = 2.0) +
    stat_summary(fun = "median", geom = "point", shape = 21, fill = "black", color = "black",
                 size = 3.5, stroke = 0.8) +
    geom_line(data = means_df, aes(x = Group_Display, y = Mean_Bias, group = Line_Group),
              color = "black", linewidth = 1) +
    geom_point(data = means_df, aes(x = Group_Display, y = Mean_Bias),
               color = "black", fill = "black", shape = 21, size = 3.5, stroke = 0.8) +
    scale_fill_manual(values = my_palette, guide = "none") +
    scale_color_manual(values = my_palette, guide = "none") +
    scale_x_discrete(labels = simple_labels) +
    annotate("text", x = 1.5, y = -1, label = "Proactive", size = 8, hjust = 0.5, family = my_font_family) +
    annotate("text", x = 4.5, y = -1, label = "Retroactive", size = 8, hjust = 0.5, family = my_font_family) +
    geom_hline(yintercept = 1, linetype = "solid", color = "black", linewidth = 0.8) +
    labs(title = toupper(corr_level), y = "narrative bias score (space < 1 < person)", x = "") +
    theme_minimal(base_size = 14) +
    theme(text = element_text(family = my_font_family, size = 25),
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent", color = NA),
          panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 1, margin = margin(t = 15), size = 22, color = "black"),
          axis.title.y = element_text(margin = margin(r = 10), size = 22),
          legend.position = "none") +
    coord_cartesian(ylim = c(-1, 11), clip = "off")
  print(p_bias)
  
  # brackets & stars with real p-values (only drawn if p < .10)
  if (corr_level == "total") {
    s1 <- star_label(p_spy_crim_pro)
    s2 <- star_label(p_crim_ctrl_pro)
    if (nzchar(s1)) p_bias <- p_bias +
      annotate("segment", x = 1, xend = 2, y = y_line_1, yend = y_line_1, color = "black", linewidth = 0.8) +
      annotate("text", x = 1.5, y = y_text_1, label = s1, size = 8, fontface = "bold")
    if (nzchar(s2)) p_bias <- p_bias +
      annotate("segment", x = 2, xend = 3, y = y_line_2, yend = y_line_2, color = "black", linewidth = 0.8) +
      annotate("text", x = 2.5, y = y_text_2, label = s2, size = 8, fontface = "bold")
  }
  print(p_bias)
  filename <- sprintf("shamem1_bias_scores_%s_nooutlier.svg", corr_level)
  ggsave(filename = file.path(PATH_OUT, filename), plot = p_bias, width = 10, height = 6, bg = "white")
  cat("Saved SVG:", filename, "\n")
  
  # --- ADDITIONAL PLOT CONTRAST PROs (star now from p_spy_crim_pro) ---
  if (corr_level == "total") {
    cat("Creating 'Proactive Only' plot for 'total' level...\n")
    d_pro_full <- d_plot %>% filter(Original_Code %in% c("Spy_Pro", "Crim_Pro", "Control"))
    control_mean <- d_pro_full %>% filter(Original_Code == "Control") %>%
      summarise(Mean_Control = mean(Bias_Score, na.rm = TRUE)) %>% pull(Mean_Control)
    d_pro <- d_pro_full %>% filter(Original_Code != "Control")
    d_pro$Group_Display <- factor(d_pro$Group_Display, levels = c("Spy_Pro", "Crim_Pro"))
    
    p_bias_pro_only <- ggplot(d_pro, aes(x = Group_Display, y = Bias_Score)) +
      geom_violin(aes(fill = Original_Code), trim = FALSE, adjust = 1.2, color = NA, alpha = 0.8,
                  quantile.colour = "black", quantile.linewidth = 0.5, quantile.linetype = "solid",
                  quantiles = c(0.25, 0.75)) +
      geom_rect(aes(xmin = x_num, xmax = x_num + 0.5, ymin = -Inf, ymax = Inf),
                fill = "white", alpha = 1, color = NA, inherit.aes = FALSE) +
      geom_point(aes(x = x_dot_jitter, color = Original_Code), alpha = 0.7, size = 2.0) +
      stat_summary(fun = "median", geom = "point", shape = 21, fill = "black", color = "black", size = 3.5, stroke = 0.8) +
      geom_hline(yintercept = control_mean, linetype = "dashed", color = "#A6A6A6", linewidth = 1.2) +
      annotate("text", x = 0.2, y = control_mean - 0.4, label = "Control",
               color = "#A6A6A6", size = 8, hjust = 0, fontface = "italic") +
      scale_fill_manual(values = my_palette, guide = "none") +
      scale_color_manual(values = my_palette, guide = "none") +
      labs(title = paste0(toupper(corr_level), " (Proactive Only)"), y = "narrative bias score", x = "") +
      theme_minimal(base_size = 14) +
      theme(text = element_text(family = my_font_family, size = 22),
            plot.background = element_rect(fill = "transparent", color = NA),
            panel.background = element_rect(fill = "transparent", color = NA),
            panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 1, margin = margin(t = 15), size = 22, color = "black"),
            axis.title.y = element_text(margin = margin(r = 10), size = 22),
            legend.position = "none") +
      coord_cartesian(ylim = c(-1, 11), clip = "off")
    
    s1_pro <- star_label(p_spy_crim_pro)   # >>> STARS FIX (same p-value as main plot)
    if (nzchar(s1_pro)) p_bias_pro_only <- p_bias_pro_only +
      annotate("segment", x = 1, xend = 2, y = y_line_1, yend = y_line_1, color = "black", linewidth = 0.8) +
      annotate("text", x = 1.5, y = y_text_1, label = s1_pro, size = 8, fontface = "bold")
    
    print(p_bias_pro_only)
    ggsave(file.path(PATH_OUT, paste0("plot_bias_", corr_level, "_proactive_only.svg")),
           p_bias_pro_only, width = 4, height = 6, dpi = 300, bg = "white")
    cat("Saved: plot_bias_", corr_level, "_proactive_only.svg\n", sep="")
  }
  
  cat("=== GLOBAL BIAS TEST (All Groups Combined) ===\n")
  global_test <- t.test(d_wide$Bias_Score, mu = 1)
  print(global_test)
  group_summary_overall <- d_wide %>%
    summarise(N = sum(!is.na(Bias_Score)), Mean = mean(Bias_Score, na.rm = TRUE),
              SD = sd(Bias_Score, na.rm = TRUE), SE = SD / sqrt(N))
  print(group_summary_overall)
  
  group_summary <- d_wide %>%
    group_by(Group_Ret) %>%
    summarise(N = sum(!is.na(Bias_Score)), Mean = mean(Bias_Score, na.rm = TRUE),
              SD = sd(Bias_Score, na.rm = TRUE), SE = SD / sqrt(N),
              t_stat = t.test(Bias_Score, mu = 1)$statistic,
              p_value = t.test(Bias_Score, mu = 1)$p.value, .groups = "drop") %>%
    arrange(Group_Ret)
  print(group_summary)
  
  group_summary$Correctness <- corr_level
  bias_summary_list[[corr_level]] <- group_summary
  
  plot_data_list[[corr_level]] <- d_wide
  
  run_robust_contrast <- function(model_obj, data_subset, g1, g2, label) {
    lvls <- levels(model_obj$model$Group_Ret)
    if(!all(c(g1, g2) %in% lvls)) { cat(label, ": Skipped (Missing levels)\n"); return(NULL) }
    vec <- setNames(rep(0, length(lvls)), lvls)
    vec[g1] <- 1; vec[g2] <- -1
    emm <- emmeans(model_obj, ~ Group_Ret)
    res <- contrast(emm, method = list(label = vec),
                    vcov = vcovHC(model_obj, type = "HC3"), adjust = "holm")
    cat(label, ":\n"); print(res)
    cat("\n=== POST-HOC Pro Dist: Bias effect of Group_Ret ===\n")
    emm_whole <- emmeans(model_obj, ~ Group_Ret)
    #print(contrast(emm_whole, method = "pairwise", adjust = "tukey"))
    print(contrast(emm_whole, method = "pairwise", adjust = "tukey",
                   vcov = vcovHC(model_obj, type = "HC3")))
  }
  
  cat("\n--- A. PRO PHASE ---\n")
  d_pro <- d_wide %>% dplyr::filter(Group_Ret %in% c("Spy_Pro", "Crim_Pro", "Control"))
  d_pro$Group_Ret <- factor(d_pro$Group_Ret)
  if(length(unique(d_pro$Group_Ret)) >= 2) {
    mod_pro <- lm(Bias_Score ~ Group_Ret, data = d_pro)
    cat("Overall Group Effect (Robust Wald Test):\n")
    print(waldtest(mod_pro, vcov = vcovHC(mod_pro, type = "HC3")))
    run_robust_contrast(mod_pro, d_pro, "Spy_Pro", "Crim_Pro", "Pro: Spy vs Criminal")
    cat(sprintf("Pro omnibus: partial eta2 = %.3f\n", partial_eta2(mod_pro)))
  } else { cat("Insufficient groups in Pro phase.\n") }
  
  cat("\n--- B. RETRO PHASE ---\n")
  d_retro <- d_wide %>% dplyr::filter(Group_Ret %in% c("Spy_Retro", "Crim_Retro", "Control"))
  d_retro$Group_Ret <- factor(d_retro$Group_Ret)
  if(length(unique(d_retro$Group_Ret)) >= 2) {
    mod_retro <- lm(Bias_Score ~ Group_Ret, data = d_retro)
    cat("Overall Group Effect (Robust Wald Test):\n")
    print(waldtest(mod_retro, vcov = vcovHC(mod_retro, type = "HC3")))
    run_robust_contrast(mod_retro, d_retro, "Spy_Retro", "Crim_Retro", "Retro: Spy vs Criminal")
    cat(sprintf("Pro omnibus: partial eta2 = %.3f\n", partial_eta2(mod_retro)))
  } else { cat("Insufficient groups in Retro phase.\n") }
  
  cat("\n--- C. EXPERIMENTAL ONLY (2x2 Interaction) ---\n")
  d_2x2 <- d_wide %>% dplyr::filter(!grepl("Control", Group_Ret))
  if(length(unique(d_2x2$Group_Ret)) == 4) {
    d_2x2 <- d_2x2 %>%
      dplyr::mutate(
        Job = factor(ifelse(grepl("Spy", Group_Ret), "Spy", "Criminal"), levels = c("Spy", "Criminal")),
        Timing = factor(ifelse(grepl("Pro", Group_Ret), "Pro", "Retro"), levels = c("Pro", "Retro")))
    mod_2x2 <- lm(Bias_Score ~ Job * Timing, data = d_2x2)
    cat("ANOVA Table (Testing Interaction):\n"); print(anova(mod_2x2))
    cat("\n=== POST-HOC ProRetro : Bias effect of Job ===\n")
    emm_whole_2x2 <- emmeans(mod_2x2, ~ Job)
    #print(contrast(emm_whole_2x2, method = "pairwise", adjust = "tukey"))
    print(contrast(emm_whole_2x2, method = "pairwise", adjust = "tukey",
                   vcov = vcovHC(mod_2x2, type = "HC3")))
    cat("\nRobust Test for Interaction (Job:Timing):\n")
    coef_name <- "JobCriminal:TimingRetro"
    if(coef_name %in% rownames(coeftest(mod_2x2, vcov = vcovHC(mod_2x2, type = "HC3")))) {
      print(coeftest(mod_2x2, vcov = vcovHC(mod_2x2, type = "HC3"))[coef_name, ])
    }
    emm_2x2 <- emmeans(mod_2x2, ~ Job | Timing)
    cat("\nSimple Effects (Spy vs Criminal) within Timing:\n")
    print(contrast(emm_2x2, method = "pairwise", adjust = "holm", vcov = vcovHC(mod_2x2, type = "HC3")))
    emm_2x2_b <- emmeans(mod_2x2, ~ Timing | Job)
    cat("\nSimple Effects (Pro vs Retro) within Job:\n")
    print(contrast(emm_2x2_b, method = "pairwise", adjust = "holm", vcov = vcovHC(mod_2x2, type = "HC3")))
  } else { cat("Missing groups for 2x2 analysis.\n") }
  
  bias_results[[corr_level]] <- list(Pro = if(exists("mod_pro")) mod_pro else NULL,
                                     Retro = if(exists("mod_retro")) mod_retro else NULL,
                                     Interaction = if(exists("mod_2x2")) mod_2x2 else NULL)
}

# Combine all updates and join to main df ONCE ---
if (length(bias_updates_list) > 0) {
  df_bias_all <- bind_rows(bias_updates_list)
  df <- df %>% dplyr::left_join(df_bias_all, by = c("Participant_ID", "Correctness"))
  cat("\nSuccessfully added 'Bias_Score' to df, aligned by Correctness.\n")
  cat("Total rows with Bias_Score:", sum(!is.na(df$Bias_Score)), "\n")
  
  if (length(bias_summary_list) > 0) {
    bias_summary_df <- bind_rows(bias_summary_list)
    write.csv(bias_summary_df,
              file.path(PATH_OUT, "bias_score_group_summaries.csv"),
              row.names = FALSE)
  }
} else {
  cat("\nNo bias scores calculated.\n")
}
#### end ####


#### Part 3b: NORMALISED BIAS SCORE (confirmatory control) ####
cat("\n================================================================\n")
cat("=== PART 3b: NORMALISED BIAS (person - spatial)/(person + spatial) ===\n")
cat("================================================================\n")

norm_results <- list()

for (corr_level in c("total", "right", "wrong")) {
  cat("\n### CORRECTNESS LEVEL:", toupper(corr_level), "###\n")
  d_sub <- df_bias_raw %>% dplyr::filter(Correctness == corr_level)
  if(nrow(d_sub) == 0) { cat("No data. Skipping.\n"); next }
  
  d_wide <- d_sub %>%
    dplyr::select(Participant_ID, Group_Ret, Category, DV_Value) %>%
    tidyr::pivot_wider(names_from = Category, values_from = DV_Value, values_fn = mean) %>%
    dplyr::filter(!is.na(person) & !is.na(spatial)) %>%
    dplyr::mutate(Bias_Norm = (person - spatial) / (person + spatial)) %>%
    filter(!Participant_ID == 57)
  
  if(nrow(d_wide) < 5) { cat("Insufficient participants. Skipping.\n"); next }
  
  cat("=== GLOBAL NORMALISED BIAS TEST vs 0 (All Groups) ===\n")
  print(t.test(d_wide$Bias_Norm, mu = 0))
  print(d_wide %>% group_by(Group_Ret) %>%
          summarise(N = sum(!is.na(Bias_Norm)), Mean = mean(Bias_Norm, na.rm = TRUE),
                    SD = sd(Bias_Norm, na.rm = TRUE), SE = SD / sqrt(N),
                    t_stat = t.test(Bias_Norm, mu = 0)$statistic,
                    p_value = t.test(Bias_Norm, mu = 0)$p.value, .groups = "drop") %>%
          arrange(Group_Ret))
  
  cat("\n--- A. PRO PHASE ---\n")
  d_pro <- d_wide %>% dplyr::filter(Group_Ret %in% c("Spy_Pro", "Crim_Pro", "Control"))
  d_pro$Group_Ret <- factor(d_pro$Group_Ret)
  if(length(unique(d_pro$Group_Ret)) >= 2) {
    mod_pro <- lm(Bias_Norm ~ Group_Ret, data = d_pro)
    cat("Overall Group Effect (Robust Wald Test):\n")
    print(waldtest(mod_pro, vcov = vcovHC(mod_pro, type = "HC3")))
    run_robust_contrast(mod_pro, d_pro, "Spy_Pro", "Crim_Pro", "Pro: Spy vs Criminal")
    cat(sprintf("Pro omnibus: partial eta2 = %.3f\n", partial_eta2(mod_pro)))
  } else { cat("Insufficient groups in Pro phase.\n") }
  
  cat("\n--- B. RETRO PHASE ---\n")
  d_retro <- d_wide %>% dplyr::filter(Group_Ret %in% c("Spy_Retro", "Crim_Retro", "Control"))
  d_retro$Group_Ret <- factor(d_retro$Group_Ret)
  if(length(unique(d_retro$Group_Ret)) >= 2) {
    mod_retro <- lm(Bias_Norm ~ Group_Ret, data = d_retro)
    cat("Overall Group Effect (Robust Wald Test):\n")
    print(waldtest(mod_retro, vcov = vcovHC(mod_retro, type = "HC3")))
    run_robust_contrast(mod_retro, d_retro, "Spy_Retro", "Crim_Retro", "Retro: Spy vs Criminal")
    cat(sprintf("Pro omnibus: partial eta2 = %.3f\n", partial_eta2(mod_retro)))
  } else { cat("Insufficient groups in Retro phase.\n") }
  
  cat("\n--- C. EXPERIMENTAL ONLY (2x2 Interaction) ---\n")
  d_2x2 <- d_wide %>% dplyr::filter(!grepl("Control", Group_Ret))
  if(length(unique(d_2x2$Group_Ret)) == 4) {
    d_2x2 <- d_2x2 %>%
      dplyr::mutate(
        Job = factor(ifelse(grepl("Spy", Group_Ret), "Spy", "Criminal"), levels = c("Spy", "Criminal")),
        Timing = factor(ifelse(grepl("Pro", Group_Ret), "Pro", "Retro"), levels = c("Pro", "Retro")))
    mod_2x2 <- lm(Bias_Norm ~ Job * Timing, data = d_2x2)
    cat("ANOVA Table (Testing Interaction):\n"); print(anova(mod_2x2))
    cat("\nRobust Test for Interaction (Job:Timing):\n")
    coef_name <- "JobCriminal:TimingRetro"
    ct <- coeftest(mod_2x2, vcov = vcovHC(mod_2x2, type = "HC3"))
    if(coef_name %in% rownames(ct)) print(ct[coef_name, ])
    cat("\nSimple Effects (Spy vs Criminal) within Timing:\n")
    print(contrast(emmeans(mod_2x2, ~ Job | Timing), method = "pairwise", adjust = "holm",
                   vcov = vcovHC(mod_2x2, type = "HC3")))
    cat("\nSimple Effects (Pro vs Retro) within Job:\n")
    print(contrast(emmeans(mod_2x2, ~ Timing | Job), method = "pairwise", adjust = "holm",
                   vcov = vcovHC(mod_2x2, type = "HC3")))
  } else { cat("Missing groups for 2x2 analysis.\n") }
  
  norm_results[[corr_level]] <- list(Pro = if(exists("mod_pro")) mod_pro else NULL,
                                     Retro = if(exists("mod_retro")) mod_retro else NULL,
                                     Interaction = if(exists("mod_2x2")) mod_2x2 else NULL)
}
#### end ####

#### Part 4: CORRELATION ATTENTIONAL AND NARRATIVE BIAS SCORE ####
cat("\n================================================================\n")
cat("=== Part 4: CORRELATION WITH EYETRACKING DATA ===\n")
cat("================================================================\n")

#### set necessary parameters (like exclusions from eyetracking data)
excluded_subs <- c(2, 3, 4, 7, 14, 16, 25, 48, 52, 58, 59, 60, 68, 74, 79, 83, 84, 85, 90, 94, 95)
correctness_applied = 'total'


#### prepare narrative data
# Apply exclusions to main dataframe (which now includes Bias_Score)
if(exists("df")) {
  df_exclude <- df %>% filter(!Participant_ID %in% excluded_subs)
  cat("Behavior data (with Bias_Score) loaded. N rows:", nrow(df_exclude), "\n")
} else {
  cat("ERROR: Main dataframe 'df' not found.\n")
  stop("Stopping script.")
}

# Filter for correctness level BEFORE joining ---
cat("\nFiltering for Correctness == correctness_applied...\n")
df_behavior_total <- df_exclude %>% 
  filter(Correctness == correctness_applied)

cat("Rows remaining after filtering for correctness_applied:", nrow(df_behavior_total), "\n")

# Check if we have valid Bias_Scores
n_valid_bias <- sum(!is.na(df_behavior_total$Bias_Score))
cat("Rows with valid Bias_Score:", n_valid_bias, "\n")

if(n_valid_bias == 0) {
  cat("ERROR: No valid Bias_Score found for 'total' level. Check previous calculation step.\n")
  stop("Stopping script.")
}

#### prepare eye tracking data
setwd(PATH_IN_eye)

# Load Eye Data
df_eye <- read.csv("ET_bias_scores_participant_block.csv")
df_eye_all <- read.csv("ET_bias_scores_participant.csv")

d_eye <- df_eye %>%
  rename(Group_Ret = "Group_Full") %>%
  mutate(
    Participant_ID = factor(trimws(as.character(Participant_ID))),
    Group_Ret = factor(Group_Ret)
  )

d_eye_all <- df_eye_all %>%
  rename(Group_Ret = "Group_Full") %>%
  mutate(
    Participant_ID = factor(trimws(as.character(Participant_ID))),
    Group_Ret = factor(Group_Ret)
  )

# Filter eye data
df_eye_exclude <- d_eye_all %>% filter(!Participant_ID %in% excluded_subs)

cat("\n--- Calculating Eye-Tracking Slopes (Block 1 to 3) ---\n")
# 1. Filter eye data for only Block 1 and Block 3
# Ensure 'block' column is numeric or consistent
d_eye_slope <- df_eye %>%
  filter(block %in% c(1, 3)) %>%
  mutate(
    Participant_ID = factor(trimws(as.character(Participant_ID))),
    block = as.numeric(as.character(block)) # Ensure block is numeric
  )

# 2. Pivot to Wide: Create columns for Block 1 and Block 3 values
d_eye_wide <- d_eye_slope %>%
  dplyr::select(Participant_ID, block, bias_fixation_count, bias_fixation_duration, bias_sample_proportion) %>%
  pivot_wider(
    names_from = block, 
    values_from = c(bias_fixation_count, bias_fixation_duration, bias_sample_proportion),
    names_sep = "_B" # e.g., bias_fixation_count_B1, bias_fixation_count_B3
  )

# 3. Calculate Slope: (Block 3 - Block 1) 
d_eye_slopes <- d_eye_wide %>%
  mutate(
    Slope_Count = bias_fixation_count_B3 - bias_fixation_count_B1,
    Slope_Duration = bias_fixation_duration_B3 - bias_fixation_duration_B1,
    Slope_Proportion = bias_sample_proportion_B3 - bias_sample_proportion_B1
  )%>%
  dplyr::select(Participant_ID, Slope_Count, Slope_Duration, Slope_Proportion)

cat("Calculated slopes for", nrow(d_eye_slopes), "participants.\n")

# 4. Apply Exclusions to the slope data too
d_eye_slopes <- d_eye_slopes %>% filter(!Participant_ID %in% excluded_subs)


#### combine data files: filtered behavior data with eye data
df_behavior_wide <- df_behavior_total %>%
  distinct(Participant_ID, Group_Ret, Bias_Score, .keep_all = FALSE)

cat("Unique participants for correlation:", nrow(df_behavior_wide), "\n")

df_combined <- inner_join(df_behavior_wide, df_eye_exclude, by = "Participant_ID")

cat("Participants after joining with Eye Data:", nrow(df_combined), "\n")

# Clean Duplicate Columns (Group_Ret.x / .y) ---
cols_x <- names(df_combined)[grep("\\.x$", names(df_combined))]
cols_y <- names(df_combined)[grep("\\.y$", names(df_combined))]

if(length(cols_x) > 0) {
  for (i in seq_along(cols_x)) {
    name_x <- cols_x[i]
    name_y <- cols_y[i]
    base_name <- gsub("\\.x$", "", name_x)
    
    if (identical(df_combined[[name_x]], df_combined[[name_y]])) {
      df_combined[[base_name]] <- df_combined[[name_x]]
      df_combined[[name_x]] <- NULL
      df_combined[[name_y]] <- NULL
    } else {
      # If not identical, try character comparison
      if (identical(as.character(df_combined[[name_x]]), as.character(df_combined[[name_y]]))) {
        df_combined[[base_name]] <- as.character(df_combined[[name_x]])
        df_combined[[name_x]] <- NULL
        df_combined[[name_y]] <- NULL
      }
    }
  }
}

# Join the calculated slopes to the main combined dataframe
df_combined <- df_combined %>%
  left_join(d_eye_slopes, by = "Participant_ID")

cat("Slope columns added. Available columns:", 
    paste(names(df_combined)[grep("Slope", names(df_combined))], collapse = ", "), "\n")

setwd(PATH_OUT)

# Helper function (exact = FALSE to handle ties)
run_spearman <- function(var1, var2, label) {
  if(sd(var1) == 0 || sd(var2) == 0) {
    cat(sprintf("-- %s: Skipped (Zero variance)\n", label))
    return(NULL)
  }
  res <- cor.test(var1, var2, method = "spearman", exact = FALSE)
  cat(sprintf("-- %s: rho = %.3f, p = %.4f\n", label, res$estimate, res$p.value))
  return(res)
}

# Store results helper
store_result <- function(res, metric, group, n) {
  if(!is.null(res)) {
    return(tibble(Group=group, Metric=metric, N=n, Rho=as.numeric(res$estimate), P_Value=res$p.value))
  }
  return(NULL)
}

#### pro ####
cat("=== PROACTIVE ===\n")
df_combined_pro <- df_combined %>% 
  drop_na(Bias_Score, bias_fixation_count, bias_fixation_duration, bias_sample_proportion) %>%
  filter(!timing == "Retro")

cat("Final N for correlation analysis:", nrow(df_combined_pro), "\n\n")

# Run Correlations ---
cat("=== CORRELATIONS (Bias_Score [Total] vs Eye-Tracking) ===\n")

res_count <- run_spearman(df_combined_pro$Bias_Score, df_combined_pro$bias_fixation_count, "Bias (Total) vs Count")
res_dur   <- run_spearman(df_combined_pro$Bias_Score, df_combined_pro$bias_fixation_duration, "Bias (Total) vs Duration")
res_prop  <- run_spearman(df_combined_pro$Bias_Score, df_combined_pro$bias_sample_proportion, "Bias (Total) vs Proportion")

# --- 5. Group-Specific Correlations ---
cat("\n=== GROUP-SPECIFIC CORRELATIONS ===\n")
correlation_results_all_pro <- list()

if("Group_Ret" %in% names(df_combined_pro)) {
  groups_to_analyze <- unique(df_combined_pro$Group_Ret)
  
  for (grp in groups_to_analyze) {
    cat(sprintf("\n--- Group: %s ---\n", grp))
    d_sub <- df_combined_pro %>% filter(Group_Ret == grp)
    
    if (nrow(d_sub) < 3) {
      cat("  Skipped: Too few participants (N < 3)\n")
      next
    }
    cat(sprintf("  N = %d\n", nrow(d_sub)))
    
    r_count <- run_spearman(d_sub$Bias_Score, d_sub$bias_fixation_count, "Count")
    r_dur   <- run_spearman(d_sub$Bias_Score, d_sub$bias_fixation_duration, "Duration")
    r_prop  <- run_spearman(d_sub$Bias_Score, d_sub$bias_sample_proportion, "Proportion")
    
    correlation_results_all_pro[[paste(grp, "c", sep="_")]] <- store_result(r_count, "Count", grp, nrow(d_sub))
    correlation_results_all_pro[[paste(grp, "d", sep="_")]] <- store_result(r_dur, "Duration", grp, nrow(d_sub))
    correlation_results_all_pro[[paste(grp, "p", sep="_")]] <- store_result(r_prop, "Proportion", grp, nrow(d_sub))
  }
  
  if(length(correlation_results_all_pro) > 0) {
    final_table <- bind_rows(correlation_results_all_pro)
    write.csv(final_table, file.path(PATH_OUT, "correlations_bias_total_by_group_pro.csv"), row.names = FALSE)
    cat("\nResults saved to: correlations_bias_total_by_group_pro.csv\n")
  }
}

# --- 6. Plotting ---
# count
plot_corr_count_pro <- ggplot(df_combined_pro, aes(x = Bias_Score, y = bias_fixation_count, color = Group_Ret, fill = Group_Ret)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", alpha = 0.2, size = 1.5, se = TRUE) +
  scale_color_manual(values = my_palette) +
  scale_fill_manual(values = my_palette) +
  labs(x = "narrative bias score (space < 1 < person)", y = "Eye-Tracking Bias (Count)") +
  theme_minimal() +
  theme(text = element_text(family = "Arial", size = 20),legend.position = "right", plot.background = element_rect(fill = "transparent", color = NA), panel.background = element_rect(fill = "transparent", color = NA))

print(plot_corr_count_pro)
ggsave(file.path(PATH_OUT, "plot_correlation_bias_count_total_pro.svg"), plot_corr_count_pro, width = 8, height = 6, dpi = 300, bg = "transparent")

# duration
plot_corr_duration_pro <- ggplot(df_combined_pro, aes(x = Bias_Score, y = bias_fixation_duration, color = Group_Ret, fill = Group_Ret)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", alpha = 0.2, size = 1.5, se = TRUE) +
  scale_color_manual(values = my_palette) +
  scale_fill_manual(values = my_palette) +
  labs(x ="narrative bias score (space < 1 < person)", y = "Eye-Tracking Bias (duration)") +
  theme_minimal() +
  theme(text = element_text(family = "Arial", size = 20),legend.position = "right", plot.background = element_rect(fill = "transparent", color = NA), panel.background = element_rect(fill = "transparent", color = NA))

print(plot_corr_duration_pro)
ggsave(file.path(PATH_OUT, "plot_correlation_bias_duration_total_pro.svg"), plot_corr_duration_pro, width = 8, height = 6, dpi = 300, bg = "transparent")

# proportion
plot_corr_proportion_pro <- ggplot(df_combined_pro, aes(x = Bias_Score, y = bias_sample_proportion, color = Group_Ret, fill = Group_Ret)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", alpha = 0.2, size = 1.5, se = TRUE) +
  scale_color_manual(values = my_palette) +
  scale_fill_manual(values = my_palette) +
  labs(x = "narrative bias score (space < 1 < person)", y = "Eye-Tracking Bias (proportion)") +
  theme_minimal() +
  theme(text = element_text(family = "Arial", size = 20),legend.position = "right", plot.background = element_rect(fill = "transparent", color = NA), panel.background = element_rect(fill = "transparent", color = NA))

print(plot_corr_proportion_pro)
ggsave(file.path(PATH_OUT, "plot_correlation_bias_proportion_total_pro.svg"), plot_corr_proportion_pro, width = 8, height = 6, dpi = 300, bg = "transparent")
cat("Plots saved.\n")

cat("\n=== CORRELATIONS: BIAS SCORE vs. EYE-TRACKING SLOPES (Block 1 -> 3) ===\n")

# Check if slope columns exist
if ("Slope_Count" %in% names(df_combined_pro)) {
  
  # Filter for valid data (Must have Bias_Score AND valid Slope data)
  df_slope_analysis <- df_combined_pro %>%
    drop_na(Bias_Score, Slope_Count, Slope_Duration, Slope_Proportion)
  
  cat("Participants with valid Bias and Slope data:", nrow(df_slope_analysis), "\n")
  
  if (nrow(df_slope_analysis) >= 3) {
    
    # 1. Global Correlations with Slopes
    cat("\n-- Global Slope Correlations --\n")
    s_count <- run_spearman(df_slope_analysis$Bias_Score, df_slope_analysis$Slope_Count, "Bias vs Slope (Count)")
    s_dur   <- run_spearman(df_slope_analysis$Bias_Score, df_slope_analysis$Slope_Duration, "Bias vs Slope (Duration)")
    s_prop  <- run_spearman(df_slope_analysis$Bias_Score, df_slope_analysis$Slope_Proportion, "Bias vs Slope (Prop)")
    
    # 2. Group-Specific Correlations with Slopes
    cat("\n-- Group-Specific Slope Correlations --\n")
    slope_results_list_pro <- list()
    
    if ("Group_Ret" %in% names(df_slope_analysis)) {
      groups_to_analyze <- unique(df_slope_analysis$Group_Ret)
      
      for (grp in groups_to_analyze) {
        d_sub <- df_slope_analysis %>% filter(Group_Ret == grp)
        if (nrow(d_sub) < 3) next
        
        cat(sprintf("\nGroup: %s (N=%d)\n", grp, nrow(d_sub)))
        
        sc <- run_spearman(d_sub$Bias_Score, d_sub$Slope_Count, "Count Slope")
        sd <- run_spearman(d_sub$Bias_Score, d_sub$Slope_Duration, "Duration Slope")
        sp <- run_spearman(d_sub$Bias_Score, d_sub$Slope_Proportion, "Prop Slope")
        
        slope_results_list_pro[[paste(grp, "sc", sep="_")]] <- store_result(sc, "Slope_Count", grp, nrow(d_sub))
        slope_results_list_pro[[paste(grp, "sd", sep="_")]] <- store_result(sd, "Slope_Duration", grp, nrow(d_sub))
        slope_results_list_pro[[paste(grp, "sp", sep="_")]] <- store_result(sp, "Slope_Proportion", grp, nrow(d_sub))
      }
      
      # Save Slope Results
      if(length(slope_results_list_pro) > 0) {
        slope_table <- bind_rows(slope_results_list_pro)
        write.csv(slope_table, file.path(PATH_OUT, "correlations_bias_vs_SLOPES_by_group_pro.csv"), row.names = FALSE)
        cat("\nSlope correlation results saved to: correlations_bias_vs_SLOPES_by_group_pro.csv\n")
      }
    }
    
    # 3. Plot Bias vs Slope
    # Helper function: Returns a CHARACTER STRING ready for parse=TRUE
    get_corr_label_str <- function(x, y) {
      test_res <- cor.test(x, y, method = "spearman", exact = FALSE)
      rho_val <- round(test_res$estimate, 3)
      p_val <- test_res$p.value
      
      # Format p-value
      p_str <- if(p_val < 0.001) "< .001" else paste("=", round(p_val, 3))
      paste0("italic(rho) == ", rho_val, " ~ ',' ~ italic(p) ~ '", p_str, "'")
    }
    
    # --- 3.1. Count Slope Plot ---
    label_count_str <- get_corr_label_str(df_slope_analysis$Bias_Score, df_slope_analysis$Slope_Count)
    
    overall_model_count <- lm(Slope_Count ~ Bias_Score, data = df_slope_analysis)
    overall_coef <- coef(overall_model_count)
    
    plot_slope_count <- ggplot(df_slope_analysis, aes(x = Bias_Score, y = Slope_Count, color = Group_Ret, fill = Group_Ret)) +
      geom_point(size = 3, alpha = 0.7) +
      geom_abline(intercept = overall_coef[1], slope = overall_coef[2], 
                  color = "black", linetype = "dashed", size = 1.2, alpha = 0.8) +
      geom_smooth(method = "lm", alpha = 0.2, size = 1.5, se = TRUE) +
      # FIX: Use label = label_count_str (a string) and parse = TRUE
      annotate("text", x = min(df_slope_analysis$Bias_Score), 
               y = max(df_slope_analysis$Slope_Count), 
               label = label_count_str, 
               hjust = 0, vjust = 1, 
               parse = TRUE, 
               size = 6, 
               color = "black") + 
      scale_color_manual(values = my_palette) +
      scale_fill_manual(values = my_palette) +
      labs(x = "narrative bias score (space < 1 < person)", y = "Δ attention bias (B3-B1, fixation count)",
           title = "Bias Score vs. Eye-Tracking Learning Slope") +
      theme_minimal() +
      theme(text = element_text(family = "Arial", size = 20),
            legend.position = "right", 
            plot.background = element_rect(fill = "transparent", color = NA), 
            panel.background = element_rect(fill = "transparent", color = NA))
    
    print(plot_slope_count)
    ggsave(file.path(PATH_OUT, "plot_correlation_bias_vs_slope_count_pro.svg"), plot_slope_count, width = 8, height = 6, dpi = 300, bg = "transparent")
    
    # --- 3.2. Duration Slope Plot ---
    label_duration_str <- get_corr_label_str(df_slope_analysis$Bias_Score, df_slope_analysis$Slope_Duration)
    overall_model_duration <- lm(Slope_Duration ~ Bias_Score, data = df_slope_analysis)
    overall_coef <- coef(overall_model_duration)
    
    plot_slope_duration <- ggplot(df_slope_analysis, aes(x = Bias_Score, y = Slope_Duration, color = Group_Ret, fill = Group_Ret)) +
      geom_point(size = 3, alpha = 0.7) +
      geom_abline(intercept = overall_coef[1], slope = overall_coef[2], 
                  color = "black", linetype = "dashed", size = 1.2, alpha = 0.8) +
      geom_smooth(method = "lm", alpha = 0.2, size = 1.5, se = TRUE) +
      annotate("text", x = min(df_slope_analysis$Bias_Score), 
               y = max(df_slope_analysis$Slope_Duration), 
               label = label_duration_str, 
               hjust = 0, vjust = 1, 
               parse = TRUE, 
               size = 6, 
               color = "black") +
      scale_color_manual(values = my_palette) +
      scale_fill_manual(values = my_palette) +
      labs(x = "narrative bias score (space < 1 < person)", y = "Δ attention bias (B3-B1, fixation duration)",
           title = "Bias Score vs. Eye-Tracking Learning Slope") +
      theme_minimal() +
      theme(text = element_text(family = "Arial", size = 20),legend.position = "right", plot.background = element_rect(fill = "transparent", color = NA), panel.background = element_rect(fill = "transparent", color = NA))
    
    print(plot_slope_duration)
    ggsave(file.path(PATH_OUT, "plot_correlation_bias_vs_slope_duration_pro.svg"), plot_slope_duration, width = 8, height = 6, dpi = 300, bg = "transparent")
    
    # --- 3.3. Proportion Slope Plot ---
    label_proportion_str <- get_corr_label_str(df_slope_analysis$Bias_Score, df_slope_analysis$Slope_Proportion)
    overall_model_proportion <- lm(Slope_Proportion ~ Bias_Score, data = df_slope_analysis)
    overall_coef <- coef(overall_model_proportion)
    
    plot_slope_proportion <- ggplot(df_slope_analysis, aes(x = Bias_Score, y = Slope_Proportion, color = Group_Ret, fill = Group_Ret)) +
      geom_point(size = 3, alpha = 0.7) +
      geom_abline(intercept = overall_coef[1], slope = overall_coef[2], 
                  color = "black", linetype = "dashed", size = 1.2, alpha = 0.8) +
      geom_smooth(method = "lm", alpha = 0.2, size = 1.5, se = TRUE) +
      annotate("text", x = min(df_slope_analysis$Bias_Score), 
               y = max(df_slope_analysis$Slope_Proportion), 
               label = label_proportion_str, 
               hjust = 0, vjust = 1, 
               parse = TRUE, 
               size = 6, 
               color = "black") +
      scale_color_manual(values = my_palette) +
      scale_fill_manual(values = my_palette) +
      labs(x = "narrative bias score (space < 1 < person)", y = "Δ attention bias (B3-B1, sample proportion)",
           title = "Bias Score vs. Eye-Tracking Learning Slope") +
      theme_minimal() +
      theme(text = element_text(family = "Arial", size = 20),legend.position = "right", plot.background = element_rect(fill = "transparent", color = NA), panel.background = element_rect(fill = "transparent", color = NA))
    
    print(plot_slope_proportion)
    ggsave(file.path(PATH_OUT, "plot_correlation_bias_vs_slope_proportion_pro.svg"), plot_slope_proportion, width = 8, height = 6, dpi = 300, bg = "transparent")
    
  } else {
    cat("Not enough participants with complete slope data for correlation.\n")
  }
} else {
  cat("Slope columns not found. Skipping slope analysis.\n")
}

#### retro ####
cat("=== RETROACTIVE ===\n")
df_combined_retro <- df_combined %>% 
  drop_na(Bias_Score, bias_fixation_count, bias_fixation_duration, bias_sample_proportion) %>%
  filter(!timing == "Pro")
  #filter(timing == "Retro")

cat("Final N for correlation analysis:", nrow(df_combined_retro), "\n\n")

# Run Correlations ---
cat("=== CORRELATIONS (Bias_Score [Total] vs Eye-Tracking) ===\n")

res_count <- run_spearman(df_combined_retro$Bias_Score, df_combined_retro$bias_fixation_count, "Bias (Total) vs Count")
res_dur   <- run_spearman(df_combined_retro$Bias_Score, df_combined_retro$bias_fixation_duration, "Bias (Total) vs Duration")
res_retrop  <- run_spearman(df_combined_retro$Bias_Score, df_combined_retro$bias_sample_proportion, "Bias (Total) vs proportion")

# --- 5. Group-Specific Correlations ---
cat("\n=== GROUP-SPECIFIC CORRELATIONS ===\n")
correlation_results_all_retro <- list()

if("Group_Ret" %in% names(df_combined_retro)) {
  groups_to_analyze <- unique(df_combined_retro$Group_Ret)
  
  for (grp in groups_to_analyze) {
    cat(sprintf("\n--- Group: %s ---\n", grp))
    d_sub <- df_combined_retro %>% filter(Group_Ret == grp)
    
    if (nrow(d_sub) < 3) {
      cat("  Skipped: Too few participants (N < 3)\n")
      next
    }
    cat(sprintf("  N = %d\n", nrow(d_sub)))
    
    r_count <- run_spearman(d_sub$Bias_Score, d_sub$bias_fixation_count, "Count")
    r_dur   <- run_spearman(d_sub$Bias_Score, d_sub$bias_fixation_duration, "Duration")
    r_retrop  <- run_spearman(d_sub$Bias_Score, d_sub$bias_sample_proportion, "proportion")
    
    correlation_results_all_retro[[paste(grp, "c", sep="_")]] <- store_result(r_count, "Count", grp, nrow(d_sub))
    correlation_results_all_retro[[paste(grp, "d", sep="_")]] <- store_result(r_dur, "Duration", grp, nrow(d_sub))
    correlation_results_all_retro[[paste(grp, "p", sep="_")]] <- store_result(r_retrop, "proportion", grp, nrow(d_sub))
  }
  
  if(length(correlation_results_all_retro) > 0) {
    final_table <- bind_rows(correlation_results_all_retro)
    write.csv(final_table, file.path(PATH_OUT, "correlations_bias_total_by_group_retro.csv"), row.names = FALSE)
    cat("\nResults saved to: correlations_bias_total_by_group_retro.csv\n")
  }
}

# --- 6. Plotting ---
# count
plot_corr_count_retro <- ggplot(df_combined_retro, aes(x = Bias_Score, y = bias_fixation_count, color = Group_Ret, fill = Group_Ret)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", alpha = 0.2, size = 1.5, se = TRUE) +
  scale_color_manual(values = my_palette) +
  scale_fill_manual(values = my_palette) +
  labs(x = "narrative bias score (space < 1 < person)", y = "Eye-Tracking Bias (Count)") +
  theme_minimal() +
  theme(text = element_text(family = "Arial", size = 20),legend.position = "right", plot.background = element_rect(fill = "transparent", color = NA), panel.background = element_rect(fill = "transparent", color = NA))

print(plot_corr_count_retro)
ggsave(file.path(PATH_OUT, "plot_correlation_bias_count_total_retro.svg"), plot_corr_count_retro, width = 8, height = 6, dpi = 300, bg = "transparent")

# duration
plot_corr_duration_retro <- ggplot(df_combined_retro, aes(x = Bias_Score, y = bias_fixation_duration, color = Group_Ret, fill = Group_Ret)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", alpha = 0.2, size = 1.5, se = TRUE) +
  scale_color_manual(values = my_palette) +
  scale_fill_manual(values = my_palette) +
  labs(x = "narrative bias score (space < 1 < person)", y = "Eye-Tracking Bias (duration)") +
  theme_minimal() +
  theme(text = element_text(family = "Arial", size = 20),legend.position = "right", plot.background = element_rect(fill = "transparent", color = NA), panel.background = element_rect(fill = "transparent", color = NA))

print(plot_corr_duration_retro)
ggsave(file.path(PATH_OUT, "plot_correlation_bias_duration_total_retro.svg"), plot_corr_duration_retro, width = 8, height = 6, dpi = 300, bg = "transparent")

# proportion
plot_corr_proportion_retro <- ggplot(df_combined_retro, aes(x = Bias_Score, y = bias_sample_proportion, color = Group_Ret, fill = Group_Ret)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", alpha = 0.2, size = 1.5, se = TRUE) +
  scale_color_manual(values = my_palette) +
  scale_fill_manual(values = my_palette) +
  labs(x = "narrative bias score (space < 1 < person)", y = "Eye-Tracking Bias (proportion)") +
  theme_minimal() +
  theme(text = element_text(family = "Arial", size = 20),legend.position = "right", plot.background = element_rect(fill = "transparent", color = NA), panel.background = element_rect(fill = "transparent", color = NA))

print(plot_corr_proportion_retro)
ggsave(file.path(PATH_OUT, "plot_correlation_bias_proportion_total_retro.svg"), plot_corr_proportion_retro, width = 8, height = 6, dpi = 300, bg = "transparent")
cat("Plots saved.\n")

cat("\n=== CORRELATIONS: BIAS SCORE vs. EYE-TRACKING SLOPES (Block 1 -> 3) ===\n")

# Check if slope columns exist
if ("Slope_Count" %in% names(df_combined_retro)) {
  
  # Filter for valid data (Must have Bias_Score AND valid Slope data)
  df_slope_analysis <- df_combined_retro %>%
    drop_na(Bias_Score, Slope_Count, Slope_Duration, Slope_Proportion)
  
  cat("Participants with valid Bias and Slope data:", nrow(df_slope_analysis), "\n")
  
  if (nrow(df_slope_analysis) >= 3) {
    
    # 1. Global Correlations with Slopes
    cat("\n-- Global Slope Correlations --\n")
    s_count <- run_spearman(df_slope_analysis$Bias_Score, df_slope_analysis$Slope_Count, "Bias vs Slope (Count)")
    s_dur   <- run_spearman(df_slope_analysis$Bias_Score, df_slope_analysis$Slope_Duration, "Bias vs Slope (Duration)")
    s_retrop  <- run_spearman(df_slope_analysis$Bias_Score, df_slope_analysis$Slope_Proportion, "Bias vs Slope (retrop)")
    
    # 2. Group-Specific Correlations with Slopes
    cat("\n-- Group-Specific Slope Correlations --\n")
    slope_results_list_retro <- list()
    
    if ("Group_Ret" %in% names(df_slope_analysis)) {
      groups_to_analyze <- unique(df_slope_analysis$Group_Ret)
      
      for (grp in groups_to_analyze) {
        d_sub <- df_slope_analysis %>% filter(Group_Ret == grp)
        if (nrow(d_sub) < 3) next
        
        cat(sprintf("\nGroup: %s (N=%d)\n", grp, nrow(d_sub)))
        
        sc <- run_spearman(d_sub$Bias_Score, d_sub$Slope_Count, "Count Slope")
        sd <- run_spearman(d_sub$Bias_Score, d_sub$Slope_Duration, "Duration Slope")
        sp <- run_spearman(d_sub$Bias_Score, d_sub$Slope_Proportion, "Prop Slope")
        
        slope_results_list_retro[[paste(grp, "sc", sep="_")]] <- store_result(sc, "Slope_Count", grp, nrow(d_sub))
        slope_results_list_retro[[paste(grp, "sd", sep="_")]] <- store_result(sd, "Slope_Duration", grp, nrow(d_sub))
        slope_results_list_retro[[paste(grp, "sp", sep="_")]] <- store_result(sp, "Slope_Proportion", grp, nrow(d_sub))
      }
      
      # Save Slope Results
      if(length(slope_results_list_retro) > 0) {
        slope_table <- bind_rows(slope_results_list_retro)
        write.csv(slope_table, file.path(PATH_OUT, "correlations_bias_vs_SLOPES_by_group_retro.csv"), row.names = FALSE)
        cat("\nSlope correlation results saved to: correlations_bias_vs_SLOPES_by_group_retro.csv\n")
      }
    }
    
    # 3. Optional: Plot Bias vs Slope
    # Count
    plot_slope_count <- ggplot(df_slope_analysis, aes(x = Bias_Score, y = Slope_Count, color = Group_Ret, fill = Group_Ret)) +
      geom_point(size = 3, alpha = 0.7) +
      geom_smooth(method = "lm", alpha = 0.2, size = 1.5, se = TRUE) +
      scale_color_manual(values = my_palette) +
      scale_fill_manual(values = my_palette) +
      labs(x = "narrative bias score (space < 1 < person)", y = "Δ attention bias (B3-B1, fixation count)",
           title = "Bias Score vs. Eye-Tracking Learning Slope") +
      theme_minimal() +
      theme(text = element_text(family = "Arial", size = 20),legend.position = "right", plot.background = element_rect(fill = "transparent", color = NA), panel.background = element_rect(fill = "transparent", color = NA))
    
    print(plot_slope_count)
    ggsave(file.path(PATH_OUT, "plot_correlation_bias_vs_slope_count_retro.svg"), plot_slope_count, width = 8, height = 6, dpi = 300, bg = "transparent")
    
    # duration
    plot_slope_duration <- ggplot(df_slope_analysis, aes(x = Bias_Score, y = Slope_Duration, color = Group_Ret, fill = Group_Ret)) +
      geom_point(size = 3, alpha = 0.7) +
      geom_smooth(method = "lm", alpha = 0.2, size = 1.5, se = TRUE) +
      scale_color_manual(values = my_palette) +
      scale_fill_manual(values = my_palette) +
      labs(x = "narrative bias score (space < 1 < person)", y = "Δ attention bias (B3-B1, fixation duration)",
           title = "Bias Score vs. Eye-Tracking Learning Slope") +
      theme_minimal() +
      theme(text = element_text(family = "Arial", size = 20),legend.position = "right", plot.background = element_rect(fill = "transparent", color = NA), panel.background = element_rect(fill = "transparent", color = NA))
    
    print(plot_slope_duration)
    ggsave(file.path(PATH_OUT, "plot_correlation_bias_vs_slope_duration_retro.svg"), plot_slope_duration, width = 8, height = 6, dpi = 300, bg = "transparent")
    
    # proportion
    plot_slope_proportion <- ggplot(df_slope_analysis, aes(x = Bias_Score, y = Slope_Proportion, color = Group_Ret, fill = Group_Ret)) +
      geom_point(size = 3, alpha = 0.7) +
      geom_smooth(method = "lm", alpha = 0.2, size = 1.5, se = TRUE) +
      scale_color_manual(values = my_palette) +
      scale_fill_manual(values = my_palette) +
      labs(x = "narrative bias score (space < 1 < person)", y = "Δ attention bias (B3-B1, sample proportion)",
           title = "Bias Score vs. Eye-Tracking Learning Slope") +
      theme_minimal() +
      theme(text = element_text(family = "Arial", size = 20),legend.position = "right", plot.background = element_rect(fill = "transparent", color = NA), panel.background = element_rect(fill = "transparent", color = NA))
    
    print(plot_slope_proportion)
    ggsave(file.path(PATH_OUT, "plot_correlation_bias_vs_slope_proportion_retro.svg"), plot_slope_proportion, width = 8, height = 6, dpi = 300, bg = "transparent")
    cat("Slope plots saved.\n")
    
  } else {
    cat("Not enough participants with complete slope data for correlation.\n")
  }
} else {
  cat("Slope columns not found. Skipping slope analysis.\n")
}
