# Example Queries for Your PDFs

Based on your processed documents, here are example queries you can use for semantic search:

## Documents in Your Database

1. **Instrument_Science_White_Paper_2021.pdf** (687 chunks) - LIGO Instrument Science White Paper
2. **T1700231-v3.pdf** (482 chunks) - LIGO Technical Note

---

## LIGO / Gravitational Wave Observatory Queries

### Instrument & Detector Technology
```bash
python ingest_pdfs.py search --query "laser interferometer design and configuration" --top_k 5
python ingest_pdfs.py search --query "detector sensitivity and noise reduction" --top_k 5
python ingest_pdfs.py search --query "mirror suspension and vibration isolation" --top_k 5
python ingest_pdfs.py search --query "optical cavity and beam path" --top_k 5
python ingest_pdfs.py search --query "calibration methods and accuracy" --top_k 5
```

### Scientific Measurements
```bash
python ingest_pdfs.py search --query "gravitational wave detection and signal processing" --top_k 5
python ingest_pdfs.py search --query "strain measurement and sensitivity curves" --top_k 5
python ingest_pdfs.py search --query "binary black hole mergers and neutron stars" --top_k 5
python ingest_pdfs.py search --query "data analysis and event detection" --top_k 5
```

### Technical Specifications
```bash
python ingest_pdfs.py search --query "frequency range and bandwidth specifications" --top_k 5
python ingest_pdfs.py search --query "power recycling and signal recycling" --top_k 5
python ingest_pdfs.py search --query "seismic noise and environmental disturbances" --top_k 5
python ingest_pdfs.py search --query "thermal noise and quantum noise limits" --top_k 5
```

### Instrument Science & Research
```bash
python ingest_pdfs.py search --query "instrument science goals and objectives" --top_k 5
python ingest_pdfs.py search --query "upgrade plans and future improvements" --top_k 5
python ingest_pdfs.py search --query "collaboration and scientific partnerships" --top_k 5
python ingest_pdfs.py search --query "measurement uncertainties and systematic errors" --top_k 5
```

---

## GAVI FCE 2.0 Proposal Queries

### Evaluation Framework
```bash
python ingest_pdfs.py search --query "full country evaluation methodology and framework" --top_k 5
python ingest_pdfs.py search --query "evaluation design and implementation approach" --top_k 5
python ingest_pdfs.py search --query "data collection and analysis methods" --top_k 5
python ingest_pdfs.py search --query "stakeholder engagement and participation" --top_k 5
```

### Capacity Building & Training
```bash
python ingest_pdfs.py search --query "capacity strengthening and technical assistance" --top_k 5
python ingest_pdfs.py search --query "training programs and skill development" --top_k 5
python ingest_pdfs.py search --query "knowledge transfer and learning hub" --top_k 5
python ingest_pdfs.py search --query "consortium expertise and technical support" --top_k 5
```

### Program Management
```bash
python ingest_pdfs.py search --query "project timeline and milestones" --top_k 5
python ingest_pdfs.py search --query "budget and resource allocation" --top_k 5
python ingest_pdfs.py search --query "monitoring and evaluation framework" --top_k 5
python ingest_pdfs.py search --query "risk management and mitigation strategies" --top_k 5
```

### Health Systems & Immunization
```bash
python ingest_pdfs.py search --query "immunization programs and vaccine delivery" --top_k 5
python ingest_pdfs.py search --query "health system strengthening" --top_k 5
python ingest_pdfs.py search --query "country health priorities and outcomes" --top_k 5
```

---

## CRS Supplier Code of Conduct Queries

### Social & Labor Standards
```bash
python ingest_pdfs.py search --query "human rights and labor standards" --top_k 5
python ingest_pdfs.py search --query "prohibition of child labor and forced labor" --top_k 5
python ingest_pdfs.py search --query "harassment and sexual exploitation prevention" --top_k 5
python ingest_pdfs.py search --query "worker safety and working conditions" --top_k 5
```

### Environmental & Governance
```bash
python ingest_pdfs.py search --query "environmental regulations and sustainability" --top_k 5
python ingest_pdfs.py search --query "corporate governance and ethical standards" --top_k 5
python ingest_pdfs.py search --query "supplier compliance and monitoring" --top_k 5
```

---

## Cross-Document Queries

### General Scientific/Technical
```bash
python ingest_pdfs.py search --query "measurement accuracy and precision" --top_k 10
python ingest_pdfs.py search --query "quality assurance and validation" --top_k 10
python ingest_pdfs.py search --query "technical specifications and requirements" --top_k 10
```

### Project Management
```bash
python ingest_pdfs.py search --query "project planning and implementation" --top_k 10
python ingest_pdfs.py search --query "stakeholder coordination and communication" --top_k 10
python ingest_pdfs.py search --query "deliverables and reporting requirements" --top_k 10
```

---

## Tips for Better Search Results

1. **Be Specific**: More specific queries tend to return better results
   - ✅ "laser interferometer calibration methods"
   - ❌ "calibration"

2. **Use Synonyms**: Try different phrasings
   - "detector sensitivity" vs "measurement precision"
   - "capacity building" vs "skill development"

3. **Combine Concepts**: Search for relationships between concepts
   - "noise reduction and detector sensitivity"
   - "training programs and capacity strengthening"

4. **Adjust top_k**: Use more results for broader topics
   ```bash
   python ingest_pdfs.py search --query "gravitational waves" --top_k 10
   ```

5. **Filter by Source**: After getting results, you can filter by filename in your code if needed

---

## Quick Test Queries

Try these to verify everything is working:

```bash
# Test LIGO documents
python ingest_pdfs.py search --query "LIGO scientific collaboration" --top_k 3

# Test GAVI document
python ingest_pdfs.py search --query "GAVI full country evaluations" --top_k 3

# Test CRS document
python ingest_pdfs.py search --query "Catholic Relief Services supplier code" --top_k 3
```

