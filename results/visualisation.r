###############################################################################
# analysis_from_json.R
# Reads raw JSON submissions and official ARC-AGI solutions directly,
# computes pixel correctness, aggregates results, does Wilcoxon tests,
# and produces a New York Times–styled table + bar plot.
###############################################################################

# 1) LOAD REQUIRED LIBRARIES
install.packages("tidyverse")  # if not already installed
install.packages("jsonlite")   # for reading JSON
install.packages("gt")         # for the NYT-styled table
install.packages("rstatix")    # optional, for easy Wilcoxon if tasks are paired

library(tidyverse)
library(jsonlite)
library(gt)
library(rstatix)   # optional, for pairwise_wilcox_test or wilcox_test

###############################################################################
# 2) CONFIGURATION: PATHS AND FILENAMES
###############################################################################
# Adjust these paths as needed to match your project structure.

results_dir <- "results"          # Where the submission_X.json files live
data_dir    <- "data/challenges"  # Where arc-agi_evaluation_solutions.json is stored

# List of JSON submissions we want to score
submission_files <- c(
  "detailed_outputs.json",
  "submission_4o.json",
  "submission_4omini.json",
  "submission_agents.json",
  "submission_agentswtool.json",
  "submission_finetune4o.json",
  "submission_finetune4omini.json"
)

# Official evaluation solutions, keyed by task_id
solutions_file <- file.path(data_dir, "arc-agi_evaluation_solutions.json")

###############################################################################
# 3) READ OFFICIAL SOLUTIONS
###############################################################################
solutions_list <- read_json(solutions_file)

# This should look something like:
# {
#   "0520fde7": [
#     [[...grid...]],  # solution for pair_index=0
#     [[...grid...]],  # solution for pair_index=1
#     ...
#   ],
#   "task_id_2": [...],
#   ...
# }

###############################################################################
# 4) FUNCTION TO COMPARE TWO GRIDS
###############################################################################
compare_grids <- function(predicted, actual) {
  # predicted, actual: each is a list of lists in R (or matrix).
  # Return a named vector: c(is_exact=BOOL, pixel_percentage=NUM, correct_count=NUM)
  
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
  # Reads a JSON submission, compares each pair's best attempt to the official solutions,
  # returns a data frame of per-task results.

  submission_name <- basename(submission_path)
  submission_data <- read_json(submission_path)
  
  # We'll build a data frame row by row:
  all_rows <- list()
  
  for (task_id in names(submission_data)) {
    # If the task_id is not in solutions_list, skip
    if (! task_id %in% names(solutions_list)) {
      next
    }
    
    pairs_submission <- submission_data[[task_id]]      # List of pair data
    pairs_solutions  <- solutions_list[[task_id]]       # Corresponding solutions
    
    # For each pair index
    for (pair_index in seq_along(pairs_submission)) {
      pair_attempts <- pairs_submission[[pair_index]]   # e.g. a named list with attempts
      if (pair_index > length(pairs_solutions)) {
        # If the solution doesn't have that many pairs, skip
        next
      }
      actual_solution <- solutions_list[[task_id]][[pair_index]]
      
      # We'll track the best attempt
      best_pixel <- 0.0
      is_solved <- 0.0
      
      # Evaluate each attempt in the pair
      for (attempt_key in names(pair_attempts)) {
        attempt_grid <- pair_attempts[[attempt_key]]
        cmp <- compare_grids(attempt_grid, actual_solution)
        # cmp is c(is_exact=?, pixel_percentage=?, correct_count=?)
        if (cmp["pixel_percentage"] > best_pixel) {
          best_pixel <- cmp["pixel_percentage"]
        }
        if (cmp["is_exact"] == 1) {
          is_solved <- 1
          # we can break here if we want to stop after finding an exact match
          break
        }
      }
      
      # Construct a row
      row_data <- tibble(
        submission_name   = submission_name,
        task_id           = task_id,
        pair_index        = pair_index - 1,  # 0-based or 1-based, your preference
        pixel_correctness = best_pixel,
        is_solved         = is_solved
      )
      all_rows <- append(all_rows, list(row_data))
    }
  }
  
  # Combine all rows
  if (length(all_rows) == 0) {
    return(tibble(
      submission_name   = character(),
      task_id           = character(),
      pair_index        = numeric(),
      pixel_correctness = numeric(),
      is_solved         = numeric()
    ))
  }
  result_df <- bind_rows(all_rows)
  result_df
}

###############################################################################
# 6) SCORE ALL SUBMISSIONS
###############################################################################
submissions_all <- tibble()

for (fname in submission_files) {
  full_path <- file.path(results_dir, fname)
  if (! file.exists(full_path)) {
    message(sprintf("File %s not found, skipping.", fname))
    next
  }
  scored_df <- score_submission(full_path, solutions_list)
  submissions_all <- bind_rows(submissions_all, scored_df)
}

# Now we have a data frame "submissions_all" with columns:
# submission_name, task_id, pair_index, pixel_correctness, is_solved

###############################################################################
# 7) MAP FILENAMES TO HUMAN-FRIENDLY NAMES
###############################################################################
submission_map <- c(
  "submission_4o.json"          = "GPT-4o",           # Good
  "submission_4omini.json"      = "GPT-4o-mini",      # Good
  "submission_agents.json"      = "Agents",
  "submission_agentswtool.json" = "Agents + Tools",
  "submission_finetune4o.json"  = "Fine-tuned GPT 4o",
  "submission_finetune4omini.json" = "Fine-tuned GPT 4o-mini",
  "detailed_outputs.json"       = "Detailed Baseline"
)

submissions_all <- submissions_all %>%
  mutate(
    submission_label = submission_map[submission_name] %>%
      coalesce(submission_name)  # fallback if not found
  )

###############################################################################
# 8) AGGREGATE RESULTS: MEAN, SD, # of tasks solved, etc.
###############################################################################
summary_df <- submissions_all %>%
  group_by(submission_label) %>%
  summarise(
    tasks_evaluated   = n_distinct(task_id),            # # of tasks
    pairs_evaluated   = n(),                            # total # of pairs
    mean_pixel        = mean(pixel_correctness, na.rm=TRUE),
    sd_pixel          = sd(pixel_correctness, na.rm=TRUE),
    se_pixel          = sd_pixel / sqrt(n()),
    solve_rate        = mean(is_solved, na.rm=TRUE) * 100  # % tasks solved
  ) %>%
  ungroup()

###############################################################################
# 9) OPTIONAL: WILCOXON TESTS IF YOU WANT TO COMPARE SUBMISSIONS
###############################################################################
# Since we have per-pair data, we can do a repeated-measures approach if each
# submission tackled the EXACT same tasks/pairs. We'll do an example:
# "pairwise" approach across all submissions. We must ensure the data is "paired."

# We'll pivot to wide format for pixel correctness if we want a direct approach.
# Then each row is (task_id, pair_index), columns are each submission label.
wide_df <- submissions_all %>%
  select(submission_label, task_id, pair_index, pixel_correctness) %>%
  pivot_wider(
    names_from = submission_label,
    values_from = pixel_correctness
  )

# If the same tasks/pairs appear for each submission, no missing data:
# Let’s do an example: Compare "GPT 4o" to each other submission with a paired Wilcoxon.
# We'll do a function for convenience:
compare_submissions <- function(df_wide, col1, col2) {
  # Keep only rows that are not NA in both columns
  df_filt <- df_wide %>%
    filter(!is.na(.data[[col1]]), !is.na(.data[[col2]]))
  
  # Paired Wilcoxon
  res <- wilcox_test(df_filt, formula = as.formula(paste0(col1, " ~ ", col2)),
                     paired=TRUE)
  res
}

# Example: let's do pairwise across summary_df$submission_label
# But we must skip the ones that don't exist. We'll keep it simple:
available_submissions <- unique(submissions_all$submission_label)
reference_sub <- "GPT 4o"  # e.g. pick GPT 4o as the reference

wilcox_results_list <- list()

if (reference_sub %in% available_submissions) {
  for (sub_label in available_submissions) {
    if (sub_label == reference_sub) next
    # do a paired test col1= reference_sub, col2=sub_label
    if (reference_sub %in% names(wide_df) && sub_label %in% names(wide_df)) {
      wres <- compare_submissions(wide_df, reference_sub, sub_label)
      wilcox_results_list[[sub_label]] <- wres
    }
  }
}

# Inspect results
wilcox_results_list

###############################################################################
# 10) CREATE A NEW YORK TIMES–STYLE TABLE
###############################################################################
nyt_table <- summary_df %>%
  select(
    submission_label,
    tasks_evaluated,
    pairs_evaluated,
    mean_pixel,
    sd_pixel,
    solve_rate
  ) %>%
  arrange(desc(mean_pixel)) %>%
  gt() %>%
  tab_header(
    title    = "ARC-AGI Evaluation Summary from JSON",
  ) %>%
  fmt_number(
    columns = c(mean_pixel, sd_pixel, solve_rate),
    decimals = 2
  ) %>%
  tab_options(
    table.font.names = "New York Times"
  )

print(nyt_table)

###############################################################################
# 11) CREATE A BAR PLOT OF MEAN PIXEL CORRECTNESS WITH REAL CI
###############################################################################
# We can compute a 95% CI = mean ± 1.96 * se_pixel.
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
    title = "Mean Pixel Correctness with 95% Confidence Intervals",
    subtitle = "Computed directly from JSON data",
    x = "Submission",
    y = "Mean Pixel Correctness (%)"
  ) +
  theme_minimal(base_size = 14)

###############################################################################
# 12) OPTIONAL: SAVE OUTPUTS
###############################################################################
# Save table as an HTML or PNG:
#gtsave(nyt_table, "nyt_table_from_json.html")
# Or:
gtsave(nyt_table, "nyt_table_from_json.png")

# Save bar plot if desired:
ggsave("barplot_pixel_correctness_from_json.png", width=6, height=4)
