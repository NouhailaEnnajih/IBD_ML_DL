# 1. Load Data
asv <- as.data.frame(readRDS("C:/Users/YLS/Documents/IBD1356/rds_outputs_IBD1356/data2.rds"))
taxa <- as.data.frame(readRDS("C:/Users/YLS/Documents/IBD1356/rds_outputs_IBD1356/taxa127.rds"))
meta <- runaccession_diagnosis  # Your metadata

# 2. Clean Genus Names
taxa$Genus[is.na(taxa$Genus)] <- "UnknownGenus"
taxa$Genus <- gsub("[^A-Za-z0-9]", "_", taxa$Genus)

# 3. Rename ASV Columns
colnames(asv) <- taxa$Genus[match(colnames(asv), rownames(taxa))]

# 4. Merge and Clean Data
asv$sample_id <- rownames(asv)
final_data <- merge(meta, asv, by.x="run_accession", by.y="sample_id")
final_data$label <- as.integer(final_data$diagnosis %in% c("UC", "CD"))
final_data <- final_data[, !names(final_data) %in% c("run_accession", "diagnosis")]

# 5. Save Results
write.csv(final_data, "IBD_data_clean.csv", row.names=FALSE)
saveRDS(final_data, "IBD_data_clean.rds")