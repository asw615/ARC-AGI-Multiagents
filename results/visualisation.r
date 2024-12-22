###############################################################################
# analysis_from_json.R
# Updated to include:
#   - Renamed labels for the NYT table
#   - Removal of solve_rate column
#   - Inclusion of Wilcoxon score and p-value from an external CSV
###############################################################################

# 1) LOAD REQUIRED LIBRARIES
# install.packages("tidyverse")  # if not already installed
# install.packages("jsonlite")   # for reading JSON
# install.packages("gt")         # for the NYT-styled table
# install.packages("rstatix")    # optional, for easy Wilcoxon if tasks are paired

library(tidyverse)
library(jsonlite)
library(gt)
library(rstatix)   # optional, for pairwise_wilcox_test or wilcox_test

###############################################################################
# 2) CONFIGURATION: PATHS AND FILENAMES
###############################################################################
results_dir <- "results"          # Where the submission_X.json files live
data_dir    <- "data/challenges"  # Where arc-agi_evaluation_solutions.json is stored

# List of JSON submissions we want to score
submission_files <- c(
  "submission_4o.json",
  "submission_4omini.json",
  "submission_agentswtool.json",
  "submission_finetune4o.json",
  "submission_finetune4omini.json"
)

# Official evaluation solutions, keyed by task_id
solutions_file <- file.path(data_dir, "arc-agi_evaluation_solutions.json")

# Location of the external CSV that contains Wilcoxon info
evaluation_summary_csv <- file.path(results_dir, "evaluation_summary.csv")

###############################################################################
# 3) READ OFFICIAL SOLUTIONS
###############################################################################
solutions_list <- read_json(solutions_file)

###############################################################################
# 4) FUNCTION TO COMPARE TWO GRIDS
###############################################################################
compare_grids <- function(predicted, actual) {
  # Basic dimension checks
  if (!is.list(predicted) || !is.list(actual)) {
    return(c(is_exact=0, pixel_percentage=0, correct_count=0))
  }
  rows <- length(actual)
  if (rows == 0) {
    return(c(is_exact=0, pixel_percentage=0, correct_count=0))
  }
  cols <- length(actual[[1]])
  
  # If dimension mismatch, automatically 0
  if (length(predicted) != rows ||
      any(sapply(predicted, length) != cols)) {
    return(c(is_exact=0, pixel_percentage=0, correct_count=0))
  }
  
  total_pixels <- rows * cols
  correct_pixels <- 0
  for (r in seq_len(rows)) {
    for (c_i in seq_len(cols)) {
      if (identical(predicted[[r]][[c_i]], actual[[r]][[c_i]])) {
        correct_pixels <- correct_pixels + 1
      }
    }
  }
  is_exact <- as.numeric(correct_pixels == total_pixels)
  pixel_pct <- (correct_pixels / total_pixels) * 100
  c(is_exact=is_exact, pixel_percentage=pixel_pct, correct_count=correct_pixels)
}

###############################################################################
# 5) FUNCTION TO SCORE A SINGLE SUBMISSION
###############################################################################
score_submission <- function(submission_path, solutions_list) {
  submission_name <- basename(submission_path)
  submission_data <- read_json(submission_path)
  
  all_rows <- list()
  
  for (task_id in names(submission_data)) {
    if (!task_id %in% names(solutions_list)) {
      next
    }
    
    pairs_submission <- submission_data[[task_id]]
    pairs_solutions  <- solutions_list[[task_id]]
    
    for (pair_index in seq_along(pairs_submission)) {
      pair_attempts <- pairs_submission[[pair_index]]
      if (pair_index > length(pairs_solutions)) {
        next
      }
      actual_solution <- pairs_solutions[[pair_index]]
      
      best_pixel <- 0.0
      is_solved <- 0.0
      
      for (attempt_key in names(pair_attempts)) {
        attempt_grid <- pair_attempts[[attempt_key]]
        cmp <- compare_grids(attempt_grid, actual_solution)
        if (cmp["pixel_percentage"] > best_pixel) {
          best_pixel <- cmp["pixel_percentage"]
        }
        if (cmp["is_exact"] == 1) {
          is_solved <- 1
          break
        }
      }
      
      row_data <- tibble(
        submission_name   = submission_name,
        task_id           = task_id,
        pair_index        = pair_index - 1,  # 0-based or 1-based
        pixel_correctness = best_pixel,
        is_solved         = is_solved
      )
      all_rows <- append(all_rows, list(row_data))
    }
  }
  
  if (length(all_rows) == 0) {
    return(tibble(
      submission_name   = character(),
      task_id           = character(),
      pair_index        = numeric(),
      pixel_correctness = numeric(),
      is_solved         = numeric()
    ))
  }
  bind_rows(all_rows)
}

###############################################################################
# 6) SCORE ALL SUBMISSIONS
###############################################################################
submissions_all <- tibble()

for (fname in submission_files) {
  full_path <- file.path(results_dir, fname)
  if (!file.exists(full_path)) {
    message(sprintf("File %s not found, skipping.", fname))
    next
  }
  scored_df <- score_submission(full_path, solutions_list)
  submissions_all <- bind_rows(submissions_all, scored_df)
}

###############################################################################
# 7) MAP FILENAMES TO HUMAN-FRIENDLY NAMES
###############################################################################
submission_map <- c(
  "submission_4o.json"             = "GPT-4o",
  "submission_4omini.json"         = "GPT-4o-mini",
  "submission_agentswtool.json"    = "Agents + Tools",
  "submission_finetune4o.json"     = "Fine-tuned GPT-4o",
  "submission_finetune4omini.json" = "Fine-tuned GPT-4o-mini"
)

submissions_all <- submissions_all %>%
  mutate(
    submission_label = submission_map[submission_name] %>%
      coalesce(submission_name)  # fallback if not found
  )

###############################################################################
# 8) AGGREGATE RESULTS: MEAN, SD, # of tasks solved, etc.
#    (Now grouped by BOTH submission_name and submission_label so we can join.)
###############################################################################
summary_df <- submissions_all %>%
  group_by(submission_name, submission_label) %>%
  summarise(
    tasks_evaluated = n_distinct(task_id),
    pairs_evaluated = n(),
    mean_pixel      = mean(pixel_correctness, na.rm = TRUE),
    sd_pixel        = sd(pixel_correctness, na.rm = TRUE),
    se_pixel        = sd_pixel / sqrt(n()),
    solve_rate      = mean(is_solved, na.rm = TRUE) * 100
  ) %>%
  ungroup()

###############################################################################
# 9) OPTIONAL: READ EXTERNAL CSV TO ADD WILCOXON INFO
###############################################################################
# The CSV file you provided includes columns:
#   submission_name, total_score, total_tasks_scored, percentage,
#   mean_pixel_correct, median_pixel_correct, wilcoxon_p_value
#
# Suppose we interpret `total_score` as the “Wilcoxon score” and
# keep `wilcoxon_p_value` as p_value. For the "Reference," we keep as-is.
###############################################################################
external_results <- read_csv(evaluation_summary_csv) %>%
  # Example of renaming for clarity:
  rename(
    tasks_solved = total_score,
    p_value        = wilcoxon_p_value
  )
# Note that for "submission_4o.json", the p_value might be "Reference" 
# or some special string. We'll keep that.

# Join by submission_name
summary_df <- left_join(summary_df, external_results, by = "submission_name")

###############################################################################
# 10) CREATE A NEW YORK TIMES–STYLE TABLE
#     - Rename columns:
#         submission_label -> model
#         pairs_evaluated  -> tasks_solved
#     - Remove solve_rate
#     - Include columns for Wilcoxon score + p-value
###############################################################################
nyt_table <- summary_df %>%
  rename(
    model        = submission_label,
    tasks = tasks_evaluated
  ) %>%
  # Select the columns you want in the final table
  select(
    model,
    tasks,
    tasks_solved,
    mean_pixel,
    sd_pixel,
    p_value  # newly joined column
  ) %>%
  arrange(desc(mean_pixel)) %>%
  gt() %>%
  tab_header(
    title = "ARC-AGI Evaluation Summary"
  ) %>%
  fmt_number(
    columns = c(mean_pixel, sd_pixel, tasks_solved),
    decimals = 2
  ) %>%
  # If p_value is numeric, you can also format it. If it might be "Reference",
  # you may need to treat that carefully. For now, just attempt numeric format:
  fmt_number(
    columns = c(p_value),
    decimals = 4
  ) %>%
  tab_options(
    table.font.names = "New York Times"
  )

print(nyt_table)

###############################################################################
# 11) CREATE A BAR PLOT OF MEAN PIXEL CORRECTNESS WITH REAL CI
###############################################################################
df_plot <- summary_df %>%
  mutate(
    ci_lower = mean_pixel - 1.96 * se_pixel,
    ci_upper = mean_pixel + 1.96 * se_pixel
  )

ggplot(df_plot, aes(x = reorder(submission_label, mean_pixel),
                    y = mean_pixel)) +
  geom_col(fill = "steelblue") +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.3) +
  coord_flip() +
  labs(
    title = "Mean Pixel Correctnesss",
    x = "",
    y = ""
  ) +
  theme_minimal(base_size = 18) +
  theme(
    plot.title = element_text(hjust = 10),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA)
  )

###############################################################################
# 12) OPTIONAL: SAVE OUTPUTS
###############################################################################
gtsave(nyt_table, "results/table_of_results.png")
ggsave("results/barplot_pixel_correctness.png", width=6, height=4)

