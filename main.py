import os
from platform import system

import numpy as np
import cv2

def clamp(val, min, max):
    if val <= min:
        return min
    elif val >= max:
        return max
    return val

def normalize_v(v, min, max):
    v[:] = np.clip(v, min, max)

def ParseImage(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def PrintErrors(error_nn, error_bilinear, error_bicubic):
    print("\nInterpolation Error Comparison:")
    print("================================")
    print("Method          | First Norm   | Frobenius Norm | Infinity Norm")
    print("---------------------------------------------------------------")
    print(f"Nearest Neighbor | {error_nn[0]:<12.4f} | {error_nn[1]:<15.4f} | {error_nn[2]:<13.4f}")
    print(f"Bilinear         | {error_bilinear[0]:<12.4f} | {error_bilinear[1]:<15.4f} | {error_bilinear[2]:<13.4f}")
    print(f"Bicubic          | {error_bicubic[0]:<12.4f} | {error_bicubic[1]:<15.4f} | {error_bicubic[2]:<13.4f}")
    print("---------------------------------------------------------------\n")

def GetNeighbors4(arr, i, j):
    nRows, nColumns, _ = np.shape(arr)
    neighbors = []

    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    for dx, dy in directions:
        ei = i + dx
        ej = j + dy

        if 0 <= ei < nRows and 0 <= ej < nColumns:
            neighbors.append(arr[ei, ej])
        else:
            neighbors.append([0, 0, 0])
    return neighbors

def GetNeighbors16(arr, i, j):
    nRows, nColumns, _ = np.shape(arr)
    neighbors = []

    directions = [(-1, -1), (-1, 0), (-1, 1), (-1, 2), (0, -1), (0, 0), (0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (1, 2), (2, -1), (2, 0), (2, 1), (2, 2)]

    for dx, dy in directions:
        ei = i + dx
        ej = j + dy

        if 0 <= ei < nRows and 0 <= ej < nColumns:
            neighbors.append(arr[ei, ej])
        else:
            neighbors.append([0, 0, 0])
    return neighbors

def BicubicInterp(arr, w, h):
    nOriginalHeight, nOriginalWidth = arr.shape[:2]
    new_image = np.zeros((h, w, 3), dtype=arr.dtype)

    flXScale = nOriginalWidth / w
    flYScale = nOriginalHeight / h

    for i in range(h):
        for j in range(w):
            flNewX = j * flXScale
            flNewY = i * flYScale
            iNewX = clamp(int(j * flXScale), 0, nOriginalWidth - 1)
            iNewY = clamp(int(i * flYScale), 0, nOriginalHeight - 1)
            vOffset = np.array([flNewX - iNewX, flNewY - iNewY])
            n = GetNeighbors16(arr, iNewY, iNewX)
            splinesX = np.array([[n[0], n[1], n[2], n[3]],
                                 [n[4], n[5], n[6], n[7]],
                                 [n[8], n[9], n[10], n[11]],
                                 [n[12], n[13], n[14], n[15]]])

            Q1 = -0.5 * vOffset + vOffset ** 2 - 0.5 * vOffset ** 3
            Q2 = 1.0 - 2.5 * vOffset ** 2 + 1.5 * vOffset ** 3
            Q3 = 0.5 * vOffset + 2.0 * vOffset ** 2 - 1.5 * vOffset ** 3
            Q4 = -0.5 * vOffset ** 2 + 0.5 * vOffset ** 3

            splinesY = np.zeros((4, 3), dtype=np.float32)

            for idx in range(0, 4):
                splinesY[idx] = splinesX[idx][0] * Q1[0] + splinesX[idx][1] * Q2[0] + splinesX[idx][2] * Q3[0] + splinesX[idx][3] * Q4[0]

            vResult = splinesY[0] * Q1[1] + splinesY[1] * Q2[1] + splinesY[2] * Q3[1] + splinesY[3] * Q4[1]
            normalize_v(vResult, 0, 255)

            new_image[i, j] = vResult

    return new_image

def BilinearInterp(arr, w, h):
    nOriginalHeight, nOriginalWidth = arr.shape[:2]
    new_image = np.zeros((h, w, 3), dtype=arr.dtype)

    flXScale = nOriginalWidth / w
    flYScale = nOriginalHeight / h

    for i in range(h):
        for j in range(w):
            flNewX = j * flXScale
            flNewY = i * flYScale
            iNewX = clamp(int(j * flXScale), 0, nOriginalWidth - 1)
            iNewY = clamp(int(i * flYScale), 0, nOriginalHeight - 1)
            flLeftColorMultiplier = flNewX - iNewX
            flRightColorMultiplier = 1 - flLeftColorMultiplier
            flTopColorMultiplier = flNewY - iNewY
            flBottomColorMultiplier = 1 - flTopColorMultiplier

            n = GetNeighbors4(arr, iNewY, iNewX)

            top, bottom = np.array(n[0]), np.array(n[1])
            right, left = np.array(n[2]), np.array(n[3])

            vAvgY = (top*flTopColorMultiplier+bottom*flBottomColorMultiplier)
            vAvgX = (right*flRightColorMultiplier+left*flLeftColorMultiplier)
            vAvg = (vAvgX + vAvgY)*0.5

            new_image[i, j] = vAvg

    return new_image


def NNInterp(arr, w, h):
    nOriginalHeight, nOriginalWidth = arr.shape[:2]
    new_image = np.zeros((h, w, 3), dtype=arr.dtype)

    flXScale = nOriginalWidth / w
    flYScale = nOriginalHeight / h

    for i in range(h):
        for j in range(w):
            iNewX = clamp(int(j * flXScale), 0, nOriginalWidth - 1)
            iNewY = clamp(int(i * flYScale), 0, nOriginalHeight - 1)

            new_image[i, j] = arr[iNewY, iNewX]

    return new_image


def CalculateError(original, interpolated):
    original = original.astype(float)
    interpolated = interpolated.astype(float)
    error_matrix = original - interpolated

    first_norm = np.sum(np.abs(error_matrix))
    frobenius_norm = np.sqrt(np.sum(error_matrix ** 2))
    inf_norm = np.max(np.abs(error_matrix))

    mse = np.mean(error_matrix ** 2)

    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse)) if mse > 0 else float('inf')
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error_matrix))

    return {
        'first_norm': first_norm,
        'frobenius_norm': frobenius_norm,
        'inf_norm': inf_norm,
        'mse': mse,
        'psnr': psnr,
        'rmse': rmse,
        'mae': mae
    }

def PrintErrors(error_nn, error_bilinear, error_bicubic):
    print("\nInterpolation Error Comparison:")
    print("=" * 80)
    metrics = ['first_norm', 'frobenius_norm', 'inf_norm', 'mse', 'psnr', 'rmse', 'mae']
    print(f"{'Method':<15} | " + " | ".join(f"{m:<12}" for m in metrics))
    print("-" * 80)

    for method, errors in [
        ("Nearest Neighbor", error_nn),
        ("Bilinear", error_bilinear),
        ("Bicubic", error_bicubic)
    ]:
        values = [f"{errors[m]:<12.4f}" for m in metrics]
        print(f"{method:<15} | " + " | ".join(values))

    print("-" * 80)

def InterpExample(path_original, path_small):
    small = ParseImage(path_small)
    original = ParseImage(path_original)

    nOriginalHeight, nOriginalWidth = original.shape[:2]
    nSmallHeight, nSmallWidth = small.shape[:2]

    dir = "./results"

    os.makedirs(dir, exist_ok=True)

    resized_nn = NNInterp(small, nOriginalWidth, nOriginalHeight)
    resized_bilinear = BilinearInterp(small, nOriginalWidth, nOriginalHeight)
    resized_bicubic = BicubicInterp(small, nOriginalWidth, nOriginalHeight)

    # cv2.imwrite(os.path.join(dir, f"og_nearest_neighbor.png"), resized_nn)
    # cv2.imwrite(os.path.join(dir, f"og_bilinear.png"), resized_bilinear)
    # cv2.imwrite(os.path.join(dir, f"og_bicubic.png"), resized_bicubic)

    cv2.imshow(f"Resized to original- Nearest Neighbor", resized_nn)
    cv2.imshow(f"Resized to original - Bilinear", resized_bilinear)
    cv2.imshow(f"Resized to original - Bicubic", resized_bicubic)

    multipliers = [2, 4, 6]

    for mult in multipliers:
        resized_nn = NNInterp(small, nSmallWidth*mult, nSmallHeight*mult)
        resized_bilinear = BilinearInterp(small, nSmallWidth*mult, nSmallHeight*mult)
        resized_bicubic = BicubicInterp(small, nSmallWidth*mult, nSmallHeight*mult)

        # cv2.imwrite(os.path.join(dir, f"bus_{mult}x_nearest_neighbor.png"), resized_nn)
        # cv2.imwrite(os.path.join(dir, f"bus_{mult}x_bilinear.png"), resized_bilinear)
        # cv2.imwrite(os.path.join(dir, f"bus_{mult}x_bicubic.png"), resized_bicubic)

        cv2.imshow(f"Resized {mult}x - Nearest Neighbor", resized_nn)
        cv2.imshow(f"Zoom {mult}x - Bilinear", resized_bilinear)
        cv2.imshow(f"Zoom {mult}x - Bicubic", resized_bicubic)


def InterpZoomExample(path_original, x, y):
    original = ParseImage(path_original)
    nOriginalHeight, nOriginalWidth = original.shape[:2]
    zoom_factors = [2, 4, 8]
    dir = "./results"

    os.makedirs(dir, exist_ok=True)

    for zoom in zoom_factors:
        slice_width = nOriginalWidth // zoom
        slice_height = nOriginalHeight // zoom
        x_start = max(0, x - slice_width // 2)
        y_start = max(0, y - slice_height // 2)
        x_start = min(x_start, nOriginalWidth - slice_width)
        y_start = min(y_start, nOriginalHeight - slice_height)

        slice = original[y_start:y_start + slice_height, x_start:x_start + slice_width]

        resized_nn = NNInterp(slice, nOriginalWidth, nOriginalHeight)
        resized_bilinear = BilinearInterp(slice, nOriginalWidth, nOriginalHeight)
        resized_bicubic = BicubicInterp(slice, nOriginalWidth, nOriginalHeight)

        cv2.imshow(f"Zoom {zoom}x - Nearest Neighbor", resized_nn)
        cv2.imshow(f"Zoom {zoom}x - Bilinear", resized_bilinear)
        cv2.imshow(f"Zoom {zoom}x - Bicubic", resized_bicubic)

        # cv2.imwrite(os.path.join(dir, f"zoom_{zoom}x_nearest_neighbor.png"), resized_nn)
        # cv2.imwrite(os.path.join(dir, f"zoom_{zoom}x_bilinear.png"), resized_bilinear)
        # cv2.imwrite(os.path.join(dir, f"zoom_{zoom}x_bicubic.png"), resized_bicubic)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

def Main():
    original = ParseImage("./images/bus_original.jpg")
    small = ParseImage("./images/bus_small.jpg")

    nOriginalHeight, nOriginalWidth = original.shape[:2]

    dir = "./results"

    # os.makedirs(dir, exist_ok=True)
    #
    resized_nn = NNInterp(small, nOriginalWidth, nOriginalHeight)
    resized_bilinear = BilinearInterp(small, nOriginalWidth, nOriginalHeight)
    resized_bicubic = BicubicInterp(small, nOriginalWidth, nOriginalHeight)

    error_nn = CalculateError(original, resized_nn)
    error_bilinear = CalculateError(original, resized_bilinear)
    error_bicubic = CalculateError(original, resized_bicubic)

    PrintErrors(error_nn, error_bilinear, error_bicubic)

    #InterpExample("./images/bus_original.jpg", "./images/bus_small.jpg")

if __name__ == '__main__':
    Main()
