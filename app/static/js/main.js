document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewBox = document.querySelector('.preview-box');
    const previewImage = document.getElementById('previewImage');
    const resetButton = document.getElementById('resetButton');
    const resultSection = document.querySelector('.result-section');
    const categoryLabel = document.querySelector('.category-label');
    const confidenceScore = document.querySelector('.confidence-score');
    const categoryDescription = document.querySelector('.category-description');
    const disposalGuide = document.querySelector('.disposal-guide');

    // 垃圾分类指南
    const categoryGuides = {
        '可回收物': {
            description: '可以回收利用的垃圾',
            guide: '清洁干燥，分类投放到对应的回收箱'
        },
        '有害垃圾': {
            description: '对人体健康或自然环境造成直接或潜在危害的垃圾',
            guide: '投放到有害垃圾收集点，不要与其他垃圾混合'
        },
        '厨余垃圾': {
            description: '日常生活中产生的食物残余',
            guide: '沥干水分，投放到厨余垃圾收集桶'
        },
        '其他垃圾': {
            description: '除可回收物、有害垃圾、厨余垃圾以外的其他生活垃圾',
            guide: '投放到其他垃圾收集桶'
        }
    };

    // 拖放处理
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary-color)';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'var(--border-color)';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border-color)';
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFile(file);
        }
    });

    // 点击上传
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    // 重置按钮
    resetButton.addEventListener('click', () => {
        dropZone.style.display = 'block';
        previewBox.hidden = true;
        resultSection.hidden = true;
        fileInput.value = '';
    });

    // 处理文件
    function handleFile(file) {
        // 显示预览
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            dropZone.style.display = 'none';
            previewBox.hidden = false;
        };
        reader.readAsDataURL(file);

        // 上传文件
        const formData = new FormData();
        formData.append('file', file);

        fetch('/api/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayResult(data.prediction);
            } else {
                throw new Error(data.error);
            }
        })
        .catch(error => {
            alert('错误：' + error.message);
        });
    }

    // 显示结果
    function displayResult(prediction) {
        const category = prediction.category_name;
        const guide = categoryGuides[category];

        categoryLabel.textContent = category;
        confidenceScore.textContent = `置信度：${prediction.confidence}%`;
        categoryDescription.textContent = guide.description;
        disposalGuide.textContent = `处理建议：${guide.guide}`;

        resultSection.hidden = false;
    }
}); 