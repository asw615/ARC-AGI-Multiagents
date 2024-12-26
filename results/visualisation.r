###############################################################################
# analysis_from_json.R
# Updated to include:
#   - Asterisks for p-values with significance codes
#   - A footnote/description about Wilcoxon test
#   - A "percentage_correct" column (formerly solve_rate)
###############################################################################

# 1) LOAD REQUIRED LIBRARIES
#install.packages("tidyverse")  # if not already installed
#install.packages("jsonlite")   # for reading JSON
#install.packages("gt")         # for the NYT-styled table
#install.packages("rstatix")    # optional, for easy Wilcoxon if tasks are paired
#install.packages("webshot2")   # for gtsave()
install.packages("webshot2")    # if not installed
webshot::install_phantomjs()    # important!

library(tidyverse)
library(jsonlite)
library(gt)
library(rstatix)   # optional, for pairwise_wilcox_test or wilcox_test
library(webshot2)  # ensure webshot2 is installed for gtsave()

###############################################################################
# 2) CONFIGURATION: PATHS AND FILENAMES
###############################################################################
results_dir <- "results"          # Where the submission_X.json files live
data_dir    <- "data/challenges"  # Where arc-agi_evaluation_solutions.json is stored

# Ensure the results directory exists
if (!dir.exists(results_dir)) {
  dir.create(results_dir)
}

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
#    Here we create 'solve_rate' as the % of tasks with is_solved=1
###############################################################################
summary_df <- submissions_all %>%
  group_by(submission_name, submission_label) %>%
  summarise(
    tasks_evaluated = n_distinct(task_id),
    pairs_evaluated = n(),
    mean_pixel      = mean(pixel_correctness, na.rm = TRUE),
    sd_pixel        = sd(pixel_correctness, na.rm = TRUE),
    se_pixel        = sd_pixel / sqrt(n()),
    solve_rate      = mean(is_solved, na.rm = TRUE) * 100,  # this is the % correct answers
    .groups = "drop"
  ) %>%
  ungroup()

###############################################################################
# 9) OPTIONAL: READ EXTERNAL CSV TO ADD WILCOXON INFO
#    Expecting columns: 
#      submission_name, total_score (= Wilcoxon rank sum?), 
#      total_tasks_scored, percentage, mean_pixel_correct, 
#      median_pixel_correct, wilcoxon_p_value
###############################################################################
external_results <- read_csv(
  evaluation_summary_csv,
  show_col_types = FALSE  # Suppress the column specification message
) %>%
  rename(
    tasks_solved = total_score,  # rename to something more human-friendly
    p_value      = wilcoxon_p_value
  )

# Verify the required columns are present
required_columns <- c("submission_name", "tasks_solved", "p_value")
missing_cols <- setdiff(required_columns, names(external_results))
if (length(missing_cols) > 0) {
  stop("Missing columns in evaluation_summary.csv: ", paste(missing_cols, collapse = ", "))
}

# Handle non-numeric p_value entries gracefully
external_results <- external_results %>%
  mutate(
    p_value = vapply(p_value, function(val) {
      val_trimmed <- trimws(val)
      if (grepl("^\\d+(\\.\\d+)?$", val_trimmed)) {
        sprintf("%.4f", as.numeric(val_trimmed))
      } else {
        val_trimmed
      }
    }, character(1))
  )

# Join by submission_name
summary_df <- left_join(summary_df, external_results, by = "submission_name")

###############################################################################
# 10) HELPER FUNCTION TO ADD ASTERISKS TO P-VALUES
###############################################################################
add_p_value_asterisks <- function(x) {
  sapply(x, function(val) {
    val_trimmed <- trimws(val)
    # If the cell isn't purely numeric (e.g. "Reference"), return as-is
    if (!grepl("^\\d+(\\.\\d+)?$", val_trimmed)) {
      return(val)
    } else {
      val_num <- as.numeric(val_trimmed)
      # Add asterisks based on common significance levels
      if (val_num < 0.001) {
        paste0(sprintf("%.4f", val_num), "***")
      } else if (val_num < 0.01) {
        paste0(sprintf("%.4f", val_num), "**")
      } else if (val_num < 0.05) {
        paste0(sprintf("%.4f", val_num), "*")
      } else {
        # No asterisks if p >= .05
        sprintf("%.4f", val_num)
      }
    }
  })
}

###############################################################################
# 11) CREATE A NEW YORK TIMES–STYLE TABLE
#     - Rename columns
#     - Include percentage of correct answers (solve_rate) in the final table
#     - Add p-value asterisks & footnotes
###############################################################################
nyt_table <- summary_df %>%
  rename(
    `Model Name` = submission_label,
    `Tasks Evaluated` = tasks_evaluated,
    `Tasks Solved` = tasks_solved,
    `Percentage Correct (%)` = solve_rate,
    `Mean Pixel Accuracy (%)` = mean_pixel,
    `Standard Deviation (Pixel)` = sd_pixel,
    `p-value` = p_value
  ) %>%
  select(
    `Model Name`,
    `Tasks Evaluated`,
    `Tasks Solved`,
    `Percentage Correct (%)`,
    `Mean Pixel Accuracy (%)`,
    `Standard Deviation (Pixel)`,
    `p-value`
  ) %>%
  arrange(desc(`Mean Pixel Accuracy (%)`)) %>%
  gt() %>%
  tab_header(
    title = "ARC-AGI Evaluation Summary"
  ) %>%
  fmt_number(
    columns = c(`Mean Pixel Accuracy (%)`, `Standard Deviation (Pixel)`, `Tasks Solved`, `Percentage Correct (%)`),
    decimals = 2
  ) %>%
  # Format p-values & add significance asterisks
  fmt(
    columns = c(`p-value`),
    fns = add_p_value_asterisks
  ) %>%
  # Add a footnote to the p_value column label
  tab_footnote(
    footnote = "p-values computed via Wilcoxon test on pixel correctness.",
    locations = cells_column_labels(columns = `p-value`)
  ) %>%
  # Add a source note below the table for significance codes
  tab_source_note(
    source_note = md("Significance codes: ***p < 0.001, **p < 0.01, *p < 0.05")
  ) %>%
  tab_options(
    table.font.names = "New York Times"
  )


###############################################################################
# 12) CREATE A BAR PLOT OF MEAN PIXEL CORRECTNESS WITH REAL CI
###############################################################################
df_plot <- summary_df %>%
  mutate(
    ci_lower = mean_pixel - 1.96 * se_pixel,
    ci_upper = mean_pixel + 1.96 * se_pixel
  )

ggplot(df_plot, aes(x = reorder(model, mean_pixel), y = mean_pixel)) +
  geom_col(fill = "steelblue") +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.3) +
  coord_flip() +
  labs(
    title = "Mean Pixel Correctness",
    x = "",
    y = ""
  ) +
  theme_minimal(base_size = 18) +
  theme(
    plot.title = element_text(hjust = 0.5),  # Center the title
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA)
  )

###############################################################################
# 13) OPTIONAL: SAVE OUTPUTS
###############################################################################
gtsave(nyt_table, "results/table_of_results.png")
#ggsave("results/barplot_pixel_correctness.png", width=6, height=4)
