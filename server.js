const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const { v4: uuidv4 } = require('uuid');
const path = require('path');

// Inisialisasi aplikasi Express
const app = express();

// Konfigurasi Multer untuk file upload (maksimum 1MB)
const upload = multer({
    limits: { fileSize: 1000000 }, // Batas ukuran file 1MB
});

// Path ke model TensorFlow.js (akses melalui WSL)
const modelPath = path.join('/mnt/e/ml-api/submissions-model', 'model.json');
let model;

// Muat model TensorFlow.js
(async () => {
    try {
        model = await tf.loadGraphModel(`file://${modelPath}`);
        console.log('Model berhasil dimuat');
    } catch (error) {
        console.error('Gagal memuat model:', error);
    }
})();

// Endpoint untuk prediksi
app.post('/predict', upload.single('image'), async (req, res) => {
    // Periksa apakah file tersedia dalam permintaan
    if (!req.file) {
        return res.status(400).json({
            status: 'fail',
            message: 'Terjadi kesalahan dalam melakukan prediksi',
        });
    }

    try {
        // Decode gambar menjadi tensor
        const buffer = req.file.buffer;
        const imageTensor = tf.node.decodeImage(buffer)
            .resizeBilinear([224, 224])
            .expandDims(0)
            .toFloat();

        // Prediksi menggunakan model
        const prediction = model.predict(imageTensor).arraySync();
        const isCancer = prediction[0][0] > 0.5; // Sesuaikan threshold jika diperlukan

        // Respons sukses dengan kode status 201
        const response = {
            status: 'success',
            message: 'Model is predicted successfully',
            data: {
                id: uuidv4(),
                result: isCancer ? 'Cancer' : 'Non-cancer',
                suggestion: isCancer
                    ? 'Segera periksa ke dokter!'
                    : 'Penyakit kanker tidak terdeteksi.',
                createdAt: new Date().toISOString(),
            },
        };

        res.status(201).json(response); // Mengembalikan status 201
    } catch (error) {
        console.error('Error during prediction:', error);
        res.status(400).json({
            status: 'fail',
            message: 'Terjadi kesalahan dalam melakukan prediksi',
        });
    }
});

// Middleware untuk menangani error terkait file terlalu besar
app.use((err, req, res, next) => {
    if (err instanceof multer.MulterError) {
        return res.status(413).json({
            status: 'fail',
            message: `Payload content length greater than maximum allowed: ${1000000}`,
        });
    }
    res.status(400).json({
        status: 'fail',
        message: 'Terjadi kesalahan dalam melakukan prediksi',
    });
});

// Jalankan server pada port 3000
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server berjalan pada port ${PORT}`);
});
