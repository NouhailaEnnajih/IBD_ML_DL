# Assuming:
# - `asv_combined` is your combined count table (samples x ASVs)
# - `taxa_combined` is your taxonomy table (ASV sequences as rownames, with a "Genus" column)

# If you previously named them differently
#it is the output from dada2
data3<- readRDS("C:/Users/YLS/Documents/autismMLandDL/rds_outputsValidation/data3_no_chimeras.rds")
asv_combined <- as.data.frame(data3)

taxa3 <- readRDS("C:/Users/YLS/Documents/autismMLandDL/rds_outputsValidation/taxa3.rds")
taxa_combined <- as.data.frame(taxa3)

##############################################
## 1. Extract and clean Genus information   ##
##############################################

# Replace NA Genus with "UnknownGenus"
genus <- taxa_combined$Genus
genus[is.na(genus)] <- "UnknownGenus"

# Clean genus names (remove special characters, spaces, etc.)
genus <- gsub(" ", "_", genus)
genus <- gsub("[^a-zA-Z0-9_]", "", genus)

# Ensure uniqueness to avoid duplicated column names
genus_unique <- make.unique(genus)

##############################################
## 2. Create a mapping between ASV and Genus ##
##############################################

# Vector to map ASV ID → Genus
name_map <- setNames(genus_unique, rownames(taxa_combined))

# Identify ASVs (columns) in count table that exist in taxonomy
cols_to_rename <- intersect(colnames(asv_combined), names(name_map))

# Rename them
colnames(asv_combined)[colnames(asv_combined) %in% cols_to_rename] <- name_map[cols_to_rename]
saveRDS(asv_combined, "asv_sequences_renamed_Korea.rds")
write.csv(asv_combined, "asv_sequences_renamed_Korea.csv")


############################### here you can merge your output with the metadata ##############################

#import the meatada

#metadata <- read.csv("C:/Users/YLS/Documents/autismMLandDL/TAXA_ASV files/your_metadata_file.csv")
metadata<-validation_metadata
asv_data<-asv_combined
# Ensure rownames of ASV table are in a column to allow merge
asv_data$run_accession <- rownames(asv_data)

# Merge with metadata using "run_accession"
merged_data <- merge(metadata, asv_data, by = "run_accession")
View(merged_data)

# Assure-toi que la colonne est bien du type caractère
merged_data$sample_alias <- as.character(merged_data$sample_alias)

# Remplacer les valeurs commençant par "A" par "1"
merged_data$sample_alias[grepl("^A", merged_data$sample_alias)] <- 1

# Remplacer celles commençant par "N" par "0"
merged_data$sample_alias[grepl("^T", merged_data$sample_alias)] <- 0
merged_data <- merged_data[, !names(merged_data) %in% "run_accession"]
write.csv(merged_data,"dataKorea_joined.csv")


names(merged_data)[names(merged_data) == "sample_alias"] <- "label"

#Save the result
write.csv(merged_data, "FinalDataKorea.csv", row.names = FALSE)
saveRDS(merged_data, "FinalDataKorea.rds")
