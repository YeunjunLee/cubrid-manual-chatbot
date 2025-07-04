CREATE TABLE docs (id INT PRIMARY KEY AUTO_INCREMENT, content STRING, embedding VECTOR(384, FLOAT));

CREATE VECTOR INDEX idx_v ON docs(embedding EUCLIDEAN)
WITH (m = 40, efConstruction = 500);

SET SYSTEM PARAMETERS 'hnsw_ef_search=500';
