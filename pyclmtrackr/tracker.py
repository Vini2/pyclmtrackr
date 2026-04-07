"""A small Python port of clmtrackr's still-image fitting path.

The implementation mirrors the original JavaScript tracker:
face-box initialization, SVM patch responses, and regularized landmark
mean-shift updates against the PCA point-distribution model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised by users without deps.
    raise ImportError(
        "pyclmtrackr requires numpy and opencv-python. "
        "Install them with `python3 -m pip install -r requirements.txt`."
    ) from exc

from .model import load_model


@dataclass
class FitResult:
    """Result returned by :meth:`CLMTracker.fit`."""

    points: np.ndarray
    parameters: np.ndarray
    bbox: tuple[int, int, int, int]
    iterations: int
    converged: bool
    movement: float


class CLMTracker:
    """Fit the default clmtrackr PCA/SVM facial model to a still image."""

    def __init__(
        self,
        model: Optional[Union[dict, str, Path]] = None,
        *,
        search_window: int = 11,
        variance_sequence: Iterable[float] = (10.0, 5.0, 1.0),
    ) -> None:
        if model is None or isinstance(model, (str, Path)):
            model = load_model(model)

        self.model = model
        self.patch_model = model["patchModel"]
        self.shape_model = model["shapeModel"]
        if self.patch_model["patchType"] != "SVM":
            raise ValueError("The Python tracker currently supports SVM patch models only.")

        self.num_patches = int(self.patch_model["numPatches"])
        self.patch_size = int(self.patch_model["patchSize"][0])
        self.search_window = int(search_window)
        self.num_parameters = int(self.shape_model["numEvalues"])
        self.mean_shape = np.asarray(self.shape_model["meanShape"], dtype=np.float64)
        self.eigen_vectors = np.asarray(self.shape_model["eigenVectors"], dtype=np.float64)
        self.eigen_values = np.asarray(self.shape_model["eigenValues"], dtype=np.float64)
        self.variance_sequence = tuple(float(v) for v in variance_sequence)
        self.non_regularized = set(self.shape_model.get("nonRegularizedVectors", []))

        weights = self.patch_model["weights"]["raw"]
        self.patch_weights = np.asarray(weights, dtype=np.float64).reshape(
            self.num_patches, self.patch_size, self.patch_size
        )
        self.patch_bias = np.asarray(self.patch_model["bias"]["raw"], dtype=np.float64)

        self.model_width = int(self.patch_model["canvasSize"][0])
        self.model_height = int(self.patch_model["canvasSize"][1])
        self.patch_width = self.patch_size + self.search_window - 1
        self.sketch_width = self.model_width + self.search_window + self.patch_size - 2
        self.sketch_height = self.model_height + self.search_window + self.patch_size - 2

        self._mean_min = self.mean_shape.min(axis=0)
        self._mean_max = self.mean_shape.max(axis=0)
        self._mean_size = self._mean_max - self._mean_min

        self._prior_base = np.zeros(self.num_parameters + 4, dtype=np.float64)
        for i, value in enumerate(self.eigen_values):
            self._prior_base[i + 4] = 1e-7 if i in self.non_regularized else 1.0 / value

    def fit(
        self,
        image: Union[str, Path, np.ndarray],
        *,
        bbox: Optional[tuple[int, int, int, int]] = None,
        max_iterations: int = 40,
        convergence_tol: float = 0.5,
        min_iterations: int = 10,
    ) -> FitResult:
        """Fit the model to an image and return 71 ``[x, y]`` landmarks.

        ``bbox`` is optional and follows OpenCV/clmtrackr convention:
        ``(x, y, width, height)``.
        """

        color = self._read_image(image)
        gray = self._to_gray(color)
        face_box = bbox or self.detect_face(gray)
        params = self._initial_parameters(gray, face_box)
        positions = self.calculate_positions(params, use_transforms=True)
        last_movement = float("inf")
        converged = False

        iteration = 0
        for iteration in range(1, max_iterations + 1):
            previous_positions = positions.copy()
            params, positions, movement = self._iterate(gray, params, positions)
            last_movement = movement
            if iteration < min_iterations:
                continue
            if movement < convergence_tol:
                converged = True
                break
            if np.linalg.norm(positions - previous_positions) < convergence_tol:
                converged = True
                break

        return FitResult(
            points=positions,
            parameters=params,
            bbox=tuple(int(v) for v in face_box),
            iterations=iteration,
            converged=converged,
            movement=float(last_movement),
        )

    def detect_face(self, gray: np.ndarray) -> tuple[int, int, int, int]:
        """Detect the largest frontal face with OpenCV's bundled Haar cascade."""

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            raise RuntimeError(f"Could not load OpenCV face cascade at {cascade_path}")

        faces = cascade.detectMultiScale(
            gray.astype(np.uint8),
            scaleFactor=1.15,
            minNeighbors=2,
            minSize=(30, 30),
        )
        if len(faces) == 0:
            raise RuntimeError("No face was detected; pass bbox=(x, y, width, height) manually.")

        return tuple(int(v) for v in max(faces, key=lambda b: b[2] * b[3]))

    def calculate_positions(self, parameters: np.ndarray, *, use_transforms: bool = True) -> np.ndarray:
        """Calculate landmark coordinates from the current CLM parameters."""

        shape_delta = (self.eigen_vectors @ parameters[4:]).reshape(self.num_patches, 2)
        shape = self.mean_shape + shape_delta
        if not use_transforms:
            return shape.copy()

        transformed = np.empty_like(shape)
        transformed[:, 0] = (parameters[0] + 1.0) * shape[:, 0] - parameters[1] * shape[:, 1] + parameters[2]
        transformed[:, 1] = parameters[1] * shape[:, 0] + (parameters[0] + 1.0) * shape[:, 1] + parameters[3]
        return transformed

    def draw(self, image: Union[str, Path, np.ndarray], result: FitResult, *, color=(0, 255, 0)) -> np.ndarray:
        """Return a copy of ``image`` with fitted landmark paths drawn on it."""

        canvas = self._read_image(image).copy()
        points = np.round(result.points).astype(np.int32)
        for path in self.model["path"]["normal"]:
            if isinstance(path, int):
                cv2.circle(canvas, tuple(points[path]), 2, color, -1, lineType=cv2.LINE_AA)
            else:
                cv2.polylines(canvas, [points[np.asarray(path, dtype=np.int32)]], False, color, 1, cv2.LINE_AA)
        return canvas

    def _iterate(
        self,
        gray: np.ndarray,
        parameters: np.ndarray,
        current_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        scale, rotation, translate_x, translate_y = self._decompose_transform(parameters)
        sketch = self._warp_to_model_space(gray, scale, rotation, translate_x, translate_y)
        patch_positions = self.calculate_positions(parameters, use_transforms=False)
        patches = self._extract_patches(sketch, patch_positions)
        responses = self._svm_responses(patches)
        original_positions = current_positions.copy()

        movement = float("inf")
        for variance in self.variance_sequence:
            jacobian = self._create_jacobian(parameters)
            mean_shift = self._mean_shift_vectors(
                responses,
                original_positions,
                current_positions,
                scale,
                variance,
            ).reshape(-1, 1)

            prior = np.diag(self._prior_base * variance)
            current = parameters.reshape(-1, 1)
            left = prior + jacobian.T @ jacobian
            right = prior @ current - jacobian.T @ mean_shift

            try:
                update = np.linalg.solve(left, right).ravel()
            except np.linalg.LinAlgError:
                update = np.linalg.lstsq(left, right, rcond=None)[0].ravel()

            old_positions = current_positions
            parameters = parameters - update
            self._clip_shape_parameters(parameters)
            current_positions = self.calculate_positions(parameters, use_transforms=True)
            movement = float(np.sum((current_positions - old_positions) ** 2))
            if movement < 1e-2:
                break

        return parameters, current_positions, movement

    def _initial_parameters(self, gray: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        eye_params = self._initial_parameters_from_eyes(gray, bbox)
        if eye_params is not None:
            return eye_params

        x, y, width, height = bbox
        scale = width / self._mean_size[1]
        rotation = 0.0
        translate_x = x - (self._mean_min[0] * scale) + 0.1 * width
        translate_y = y - (self._mean_min[1] * scale) + 0.25 * height

        parameters = np.zeros(self.num_parameters + 4, dtype=np.float64)
        parameters[0] = scale * np.cos(rotation) - 1.0
        parameters[1] = scale * np.sin(rotation)
        parameters[2] = translate_x
        parameters[3] = translate_y
        return parameters

    def _initial_parameters_from_eyes(
        self,
        gray: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        x, y, width, height = bbox
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            return None

        top = y
        bottom = y + int(0.65 * height)
        roi = gray[top:bottom, x : x + width].astype(np.uint8)
        eyes = cascade.detectMultiScale(
            roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(max(8, width // 12), max(8, height // 12)),
        )
        pair = self._select_eye_pair(eyes)
        if pair is None:
            return None

        centers = []
        for ex, ey, ew, eh in pair:
            centers.append(np.asarray([x + ex + ew / 2.0, top + ey + eh / 2.0], dtype=np.float64))
        centers.sort(key=lambda point: point[0])
        left_eye, right_eye = centers

        model_left = np.asarray(self.model["hints"]["leftEye"], dtype=np.float64)
        model_right = np.asarray(self.model["hints"]["rightEye"], dtype=np.float64)
        model_delta = model_right - model_left
        image_delta = right_eye - left_eye
        model_distance = float(np.linalg.norm(model_delta))
        image_distance = float(np.linalg.norm(image_delta))
        if model_distance <= 1e-12 or image_distance <= 1e-12:
            return None

        scale = image_distance / model_distance
        rotation = float(np.arctan2(image_delta[1], image_delta[0]) - np.arctan2(model_delta[1], model_delta[0]))
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rotated_left = scale * np.asarray(
            [
                cos_r * model_left[0] - sin_r * model_left[1],
                sin_r * model_left[0] + cos_r * model_left[1],
            ]
        )
        translate_x, translate_y = left_eye - rotated_left

        parameters = np.zeros(self.num_parameters + 4, dtype=np.float64)
        parameters[0] = scale * cos_r - 1.0
        parameters[1] = scale * sin_r
        parameters[2] = translate_x
        parameters[3] = translate_y
        return parameters

    @staticmethod
    def _select_eye_pair(eyes: np.ndarray) -> Optional[list[np.ndarray]]:
        if len(eyes) < 2:
            return None

        best_pair = None
        best_score = -float("inf")
        for i in range(len(eyes)):
            for j in range(i + 1, len(eyes)):
                first = eyes[i]
                second = eyes[j]
                c1 = np.asarray([first[0] + first[2] / 2.0, first[1] + first[3] / 2.0])
                c2 = np.asarray([second[0] + second[2] / 2.0, second[1] + second[3] / 2.0])
                separation = abs(c2[0] - c1[0])
                vertical_gap = abs(c2[1] - c1[1])
                if separation < max(first[2], second[2]):
                    continue
                score = separation - 1.5 * vertical_gap + 0.01 * (first[2] * first[3] + second[2] * second[3])
                if score > best_score:
                    best_pair = [first, second]
                    best_score = score
        return best_pair

    def _decompose_transform(self, parameters: np.ndarray) -> tuple[float, float, float, float]:
        scale = float(np.hypot(parameters[0] + 1.0, parameters[1]))
        rotation = float(np.arctan2(parameters[1], parameters[0] + 1.0))
        return scale, rotation, float(parameters[2]), float(parameters[3])

    def _warp_to_model_space(
        self,
        gray: np.ndarray,
        scale: float,
        rotation: float,
        translate_x: float,
        translate_y: float,
    ) -> np.ndarray:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        matrix = np.asarray(
            [
                [scale * cos_r, -scale * sin_r, translate_x],
                [scale * sin_r, scale * cos_r, translate_y],
            ],
            dtype=np.float64,
        )
        return cv2.warpAffine(
            gray,
            matrix,
            (self.sketch_width, self.sketch_height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float32)

    def _extract_patches(self, sketch: np.ndarray, patch_positions: np.ndarray) -> np.ndarray:
        patches = []
        size = (self.patch_width, self.patch_width)
        for x, y in patch_positions:
            center = (float(x), float(y))
            patch = cv2.getRectSubPix(sketch, size, center)
            patches.append(patch.astype(np.float64))
        return np.stack(patches, axis=0)

    def _svm_responses(self, patches: np.ndarray) -> np.ndarray:
        responses = np.empty((self.num_patches, self.search_window, self.search_window), dtype=np.float64)
        for i, patch in enumerate(patches):
            patch = self._normalize_patch(patch)
            response = cv2.matchTemplate(
                patch.astype(np.float32),
                self.patch_weights[i].astype(np.float32),
                cv2.TM_CCORR,
            ).astype(np.float64)
            response += self.patch_bias[i]
            response = 1.0 / (1.0 + np.exp(-(response - 1.0)))
            responses[i] = self._normalize_response(response)
        return responses

    def _mean_shift_vectors(
        self,
        responses: np.ndarray,
        original_positions: np.ndarray,
        current_positions: np.ndarray,
        scale: float,
        variance: float,
    ) -> np.ndarray:
        vectors = np.zeros((self.num_patches, 2), dtype=np.float64)
        half = (self.search_window - 1) * scale / 2.0

        for point_index in range(self.num_patches):
            origin_x = original_positions[point_index, 0] - half
            origin_y = original_positions[point_index, 1] - half
            weighted_sum = np.zeros(2, dtype=np.float64)
            probability_sum = 0.0

            for row in range(self.search_window):
                y = origin_y + row * scale
                for col in range(self.search_window):
                    x = origin_x + col * scale
                    dx = current_positions[point_index, 0] - x
                    dy = current_positions[point_index, 1] - y
                    probability = responses[point_index, row, col] * np.exp(
                        -0.5 * ((dx * dx) + (dy * dy)) / (variance * scale)
                    )
                    probability_sum += probability
                    weighted_sum += probability * np.asarray([x, y])

            if probability_sum > 1e-12:
                target = weighted_sum / probability_sum
                vectors[point_index] = target - current_positions[point_index]

        return vectors

    def _create_jacobian(self, parameters: np.ndarray) -> np.ndarray:
        jacobian = np.zeros((2 * self.num_patches, self.num_parameters + 4), dtype=np.float64)
        shaped_delta = (self.eigen_vectors @ parameters[4:]).reshape(self.num_patches, 2)
        shaped = self.mean_shape + shaped_delta

        for i in range(self.num_patches):
            x, y = shaped[i]
            row_x = 2 * i
            row_y = row_x + 1

            jacobian[row_x, 0] = x
            jacobian[row_y, 0] = y
            jacobian[row_x, 1] = -y
            jacobian[row_y, 1] = x
            jacobian[row_x, 2] = 1.0
            jacobian[row_y, 3] = 1.0

            ev_x = self.eigen_vectors[row_x]
            ev_y = self.eigen_vectors[row_y]
            jacobian[row_x, 4:] = (parameters[0] + 1.0) * ev_x - parameters[1] * ev_y
            jacobian[row_y, 4:] = parameters[1] * ev_x + (parameters[0] + 1.0) * ev_y

        return jacobian

    def _clip_shape_parameters(self, parameters: np.ndarray) -> None:
        limits = 3.0 * np.sqrt(self.eigen_values)
        parameters[4:] = np.clip(parameters[4:], -limits, limits)

    @staticmethod
    def _normalize_patch(patch: np.ndarray) -> np.ndarray:
        minimum = float(patch.min())
        maximum = float(patch.max())
        distance = maximum - minimum
        if distance <= 1e-12:
            return np.zeros_like(patch, dtype=np.float64)
        return (patch - minimum) / distance

    @staticmethod
    def _normalize_response(response: np.ndarray) -> np.ndarray:
        minimum = float(response.min())
        maximum = float(response.max())
        distance = maximum - minimum
        if distance <= 1e-12:
            return response
        return (response - minimum) / distance

    @staticmethod
    def _read_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            data = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if data is None:
                raise FileNotFoundError(f"Could not read image: {image}")
            return data
        if image.ndim == 2:
            return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return image

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.astype(np.float32)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
