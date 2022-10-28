"""Automated Rectification of Image.
References
----------
1.  Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.
    "Auto-rectification of user photos." 2014 IEEE International Conference on
    Image Processing (ICIP). IEEE, 2014.
2.  Bazin, Jean-Charles, and Marc Pollefeys. "3-line RANSAC for orthogonal
    vanishing point detection." 2012 IEEE/RSJ International Conference on
    Intelligent Robots and Systems. IEEE, 2012.
"""
import sys
import numpy as np
import logging
import cv2
import matplotlib.pyplot as plt
import io
import base64


def compute_edgelets(image, sigma=3):
    """Create edgelets as in the paper.
    Uses canny edge detection and then finds (small) lines using probabilstic
    hough transform as edgelets.
    Parameters
    ----------
    image: ndarray
        Image for which edgelets are to be computed.
    sigma: float
        Smoothing to be used for canny edge detection.
    Returns
    -------
    locations: ndarray of shape (n_edgelets, 2)
        Locations of each of the edgelets.
    directions: ndarray of shape (n_edgelets, 2)
        Direction of the edge (tangent) at each of the edgelet.
    strengths: ndarray of shape (n_edgelets,)
        Length of the line segments detected for the edgelet.
    """

    img_h, img_w, _ = image.shape
    locations = []
    directions = []
    strengths = []

    # First Line Detectorを使う
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fld = cv2.ximgproc.createFastLineDetector()
    linesFLD = fld.detect(gray_img)

    for line in linesFLD:
        x0, y0, x1, y1 = map(int, line[0])
        p0, p1 = np.array((x0,y0)), np.array((x1,y1))
        locations.append((p0 + p1) / 2)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # # Line Segment Detectorを使う
    # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_img_equalized = cv2.equalizeHist(gray_img) # なくてもいい
    # linesLSD = lsd(gray_img)
    # for line in linesLSD:
    #     x0, y0, x1, y1 = map(int,line[:4])
    #     p0, p1 = np.array((x0,y0)), np.array((x1,y1))
    #     locations.append((p0 + p1) / 2)
    #     directions.append(p1 - p0)
    #     strengths.append(np.linalg.norm(p1 - p0))

    # # 確率的ハフ変換を使う
    # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB2〜 でなく BGR2〜 を指定
    # gray_img_equalized = cv2.equalizeHist(gray_img)
    # gray_img = np.hstack((gray_img_equalized,equ)) #stacking images side-by-side
    # edges =cv2.Canny(gray_img, 100, 200, sigma)
    # img_s = img_h * img_w / 100000 # エッジの長さを面積とヒューリスティックに決めた定数で正規化する
    # linesPHough = cv2.HoughLinesP(gray_img, rho=1, theta=np.pi/360, threshold=80, minLineLength=int(6*img_s), maxLineGap=int(1.3*img_s))
    # for p0, p1 in linesPHough:
    #     p0, p1 = np.array(p0), np.array(p1)
    #     locations.append((p0 + p1) / 2)
    #     directions.append(p1 - p0)
    #     strengths.append(np.linalg.norm(p1 - p0))

    # convert to numpy arrays and normalize
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)

    directions = np.array(directions) / np.linalg.norm(directions, axis=1)[:, np.newaxis]

    return (locations, directions, strengths)


def edgelet_lines(edgelets):
    """Compute lines in homogenous system for edglets.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    Returns
    -------
    lines: ndarray of shape (n_edgelets, 3)
        Lines at each of edgelet locations in homogenous system.
    """
    locations, directions, _ = edgelets
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines


def compute_votes(edgelets, model, threshold_inlier=5):
    """Compute votes for each of the edgelet against a given vanishing point.
    Votes for edgelets which lie inside threshold are same as their strengths,
    otherwise zero.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    model: ndarray of shape (3,)
        Vanishing point model in homogenous cordinate system.
    threshold_inlier: float
        Threshold to be used for computing inliers in degrees. Angle between
        edgelet direction and line connecting the  Vanishing point model and
        edgelet location is used to threshold.
    Returns
    -------
    votes: ndarry of shape (n_edgelets,)
        Votes towards vanishing point model for each of the edgelet.
    """

    locations, directions, strengths = edgelets

    if model is None:
        return np.zeros_like(strengths)

    vp = model[:2] / model[2]

    est_directions = locations - vp
    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = dot_prod / abs_prod
    cosine_theta[np.where(cosine_theta<-1)] = -1
    cosine_theta[np.where(cosine_theta>1)] = 1

    theta = np.arccos(np.abs(cosine_theta))
    theta_thresh = np.deg2rad(threshold_inlier)

    lambda_ = 0.1
    votes_values = np.ones_like(theta)
    votes_values -= np.exp(-lambda_ * np.cos(theta)**2 )
    votes_values /= 1 - np.exp(-lambda_)

    theta_from_vp = np.arctan2(est_directions[:,1], est_directions[:,0])

    is_inlier = np.where((theta<=theta_thresh), 1, 0)

    # 消失点がエッジ上に存在する状況はおかしいので，そのエッジからの投票は0にする
    # 「エッジを延長した直線」と「消失点-エッジ中央点を結ぶ直線」のなす角度が1度以下かつ
    # 「消失点-エッジ中央点を結ぶ線分」の長さがエッジの長さの半分より短い時，消失点がエッジ上に存在すると判定
    is_not_vp_on_the_edge = np.where((theta<=np.deg2rad(1)) & (np.linalg.norm(est_directions, axis=1) <= strengths/2), 0, 1)

    # 元論文に即すと以下の返り値になる。角度に依存。
    # return is_vote_list * votes_values

    # 元の実装コードでは以下の返り値だった。エッジの長さに依存。
    # return is_vote_list * strengths

    # エッジの長さと角度の両方に依存する値に改良。
    return strengths * is_inlier * is_not_vp_on_the_edge * votes_values

def ransac_vanishing_point(edgelets, num_ransac_iter=2000, threshold_inlier=5):
    """Estimate vanishing point using Ransac.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    num_ransac_iter: int
        Number of iterations to run ransac.
    threshold_inlier: float
        threshold to be used for computing inliers in degrees.
    Returns
    -------
    best_model: ndarry of shape (3,)
        Best model for vanishing point estimated.
    Reference
    ---------
    Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.
    "Auto-rectification of user photos." 2014 IEEE International Conference on
    Image Processing (ICIP). IEEE, 2014.
    """
    locations, directions, strengths = edgelets
    lines = edgelet_lines(edgelets)

    num_pts = strengths.size

    # firstとsecondで被る確率を下げるため，firstは上位20%から，secondは上位50%から選ぶ
    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_vanishing_point = None
    best_votes = np.zeros(num_pts)

    for ransac_iter in range(num_ransac_iter):
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_vanishing_point = np.cross(l1, l2)
        current_vanishing_point *= np.sign(current_vanishing_point[2])

        if np.sum(current_vanishing_point**2) < 1 or current_vanishing_point[2] == 0:
            # reject degenerate candidates
            continue

        current_votes = compute_votes(edgelets, current_vanishing_point, threshold_inlier)

        if current_votes.sum() > best_votes.sum():
            best_vanishing_point = current_vanishing_point
            best_votes = current_votes
            logging.info("Current best model has {} votes at iteration {}".format(
                current_votes.sum(), ransac_iter))

    return best_vanishing_point


def reestimate_model(model, edgelets, threshold_reestimate=5):
    """Reestimate vanishing point using inliers and least squares.
    All the edgelets which are within a threshold are used to reestimate model
    Parameters
    ----------
    model: ndarry of shape (3,)
        Vanishing point model in homogenous coordinates which is to be
        reestimated.
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
        All edgelets from which inliers will be computed.
    threshold_inlier: float
        threshold to be used for finding inlier edgelets.
    Returns
    -------
    restimated_model: ndarry of shape (3,)
        Reestimated model for vanishing point in homogenous coordinates.
    """
    locations, directions, strengths = edgelets

    inliers = compute_votes(edgelets, model, threshold_reestimate) > 0
    locations = locations[inliers]
    directions = directions[inliers]
    strengths = strengths[inliers]

    lines = edgelet_lines((locations, directions, strengths))

    a = lines[:, :2]
    b = -lines[:, 2]
    est_model = np.linalg.lstsq(a, b)[0]
    return np.concatenate((est_model, [1.]))


def remove_inliers(model, edgelets, threshold_inlier=10):
    """Remove all inlier edglets of a given model.
    Parameters
    ----------
    model: ndarry of shape (3,)
        Vanishing point model in homogenous coordinates which is to be
        reestimated.
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    threshold_inlier: float
        threshold to be used for finding inlier edgelets.
    Returns
    -------
    edgelets_new: tuple of ndarrays
        All Edgelets except those which are inliers to model.
    """
    inliers = compute_votes(edgelets, model, 10) > 0
    locations, directions, strengths = edgelets
    locations = locations[~inliers]
    directions = directions[~inliers]
    strengths = strengths[~inliers]
    edgelets = (locations, directions, strengths)
    return edgelets


def compute_homography_and_warp(image, vp_h, vp_v):
    """Compute homography from vanishing points and warp the image.
    It is assumed that vp1 and vp2 correspond to horizontal and vertical
    directions, although the order is not assumed.
    Firstly, projective transform is computed to make the vanishing points go
    to infinty so that we have a fronto parellel view. Then,Computes affine
    transfom  to make axes corresponding to vanishing points orthogonal.
    Finally, Image is translated so that the image is not missed. Note that
    this image can be very large. `clip` is provided to deal with this.
    Parameters
    ----------
    image: ndarray
        Image which has to be wrapped.
    vp1: ndarray of shape (3, )
        First vanishing point in homogenous coordinate system.
    vp2: ndarray of shape (3, )
        Second vanishing point in homogenous coordinate system.
    clip: bool, optional
        If True, image is clipped to clip_factor.
    clip_factor: float, optional
        Proportion of image in multiples of image size to be retained if gone
        out of bounds after homography.
    Returns
    -------
    warped_img: ndarray
        Image warped using homography as described above.
    """
    # Find Projective Transform
    img_h, img_w, _ = image.shape

    # 垂直消失点がy軸負の方向に存在する場合は画像と消失点を上に平行移動して処理して元に戻す
    if vp_v[1]/vp_v[2]<0:
        vp_v[1] -= img_h*vp_v[2]

    vanishing_line = np.cross(vp_h, vp_v)
    H = np.eye(3)
    H[2] = vanishing_line / vanishing_line[2]
    H = H / H[2, 2]

    # Find directions corresponding to vanishing points
    v_post_h = np.dot(H, vp_h)
    v_post_v = np.dot(H, vp_v)
    v_post_h = v_post_h / np.sqrt(v_post_h[0]**2 + v_post_h[1]**2)
    v_post_v = v_post_v / np.sqrt(v_post_v[0]**2 + v_post_v[1]**2)

    if v_post_v[1] < 0:
        v_post_v = -v_post_v
        v_post_h = -v_post_h

    Shear = np.array([[v_post_h[0], v_post_v[0], 0], # x座標
                      [v_post_h[1], v_post_v[1], 0], # y座標
                      [0, 0, 1]])

    # Might be a reflection. If so, remove reflection.
    if np.linalg.det(Shear) < 0:
        Shear[:, 0] = -Shear[:, 0]

    Shear_inv = np.linalg.inv(Shear)

    # Translate so that whole of the image is covered
    inter_matrix = np.dot(Shear_inv, H)

    TODO: autoもしくはhorizontalのとき平行消失点がx軸負の方向に存在した場合の処理を追加
    # 垂直消失点がy軸負の方向に存在する場合は画像と消失点を上に平行移動してホモグラフィ変換し，その後もとに戻す
    if vp_v[1]/vp_v[2]<0:

        T_up = np.array([[1, 0, 0],
                         [0, 1, -img_h],
                         [0, 0, 1]])

        T_down = np.array([[1, 0, 0],
                           [0, 1, img_h],
                           [0, 0, 1]])

        inter_matrix = np.dot(inter_matrix, T_up)
        inter_matrix = np.dot(T_down, inter_matrix)

    # Homography変換により生じる幅や高さの変化を考慮した平行移動
    img_corners = np.array([[0, 0, img_w, img_w],
                            [0, img_h, 0, img_h],
                            [1, 1, 1, 1]])

    warped_img_corners = np.dot(inter_matrix, img_corners)
    warped_img_corners = warped_img_corners[:2] / warped_img_corners[2]

    x_offset = min(0, warped_img_corners[0].min())
    y_offset = min(0, warped_img_corners[1].min())

    max_x = int(warped_img_corners[0].max() - x_offset)
    max_y = int(warped_img_corners[1].max() - y_offset)

    T = np.array([[1, 0, -x_offset],
                  [0, 1, -y_offset],
                  [0, 0, 1]])

    warp_homography = np.dot(T, inter_matrix)

    warped_img = cv2.warpPerspective(image, warp_homography,(max_x, max_y),flags=cv2.INTER_LANCZOS4)

    return warped_img, warp_homography

def find_vanishing_points(image, num_ransac_iter, reestimate):
    # Compute all edgelets.
    edgelets1 = compute_edgelets(image)

    # image_shape = image.shape
    img_h, img_w, _ = image.shape

    threshold_inlier_degree = 2

    # Find first vanishing point
    vp1 = ransac_vanishing_point(edgelets1, num_ransac_iter, threshold_inlier_degree)
    if vp1 is None:
        print(f'Vanishing point 1st is not found.')
        vp1 = [0, 0, 0]
    if reestimate:
        vp1 = reestimate_model(vp1, edgelets1, threshold_inlier_degree)
    # Remove inlier to remove dominating direction.
    edgelets2 = remove_inliers(vp1, edgelets1, threshold_inlier_degree)

    # Find second vanishing point
    vp2 = ransac_vanishing_point(edgelets2, num_ransac_iter, threshold_inlier_degree)
    if vp2 is None:
        print(f'Vanishing point 2nd is not found.')
        vp2 = [0, 0, 0]
    if reestimate:
        vp2 = reestimate_model(vp2, edgelets2, threshold_inlier_degree)
    # Remove inlier to remove dominating direction.
    edgelets3 = remove_inliers(vp2, edgelets2, threshold_inlier_degree)

    # Find third vanishing point
    vp3 = ransac_vanishing_point(edgelets3, num_ransac_iter, threshold_inlier_degree)
    if vp3 is None:
        print(f'Vanishing point 3rd is not found.')
        vp3 = [0, 0, 0]
    if reestimate:
        vp3 = reestimate_model(vp3, edgelets3, threshold_inlier_degree)

    # isShow = True
    # visualization.vis_edgelets(image, edgelets1, isShow, '検出された全てのエッジ')
    # visualization.vis_vanishing_point(image, vp1, isShow, '消失点推定結果1')
    # visualization.vis_vanishing_point(image, vp2, isShow, '消失点推定結果2')
    # visualization.vis_vanishing_point(image, vp3, isShow, '消失点推定結果3')

    return [vp1, vp2, vp3]


def classify_vanishing_points(img_h, img_w, vp_list, INF, limit_v=10, limit_h=15, align='vertial'):

    is_limited = True if limit_v > 0 else False

    # 消失点の画像中心からの距離をx,yに分けて消失点のタイプを分類
    vp_abs_norm_x_list = np.array([np.abs(vp[0]/vp[2] - img_w/2)/(img_w/2) for vp in vp_list])
    vp_abs_norm_y_list = np.array([np.abs(vp[1]/vp[2] - img_h/2)/(img_h/2) for vp in vp_list])
    vp_norm_x_list = np.array([np.abs(vp[0]/vp[2] - img_w/2)/(img_w/2) for vp in vp_list])

    is_horizontal_vp_candidate =  np.where((vp_abs_norm_x_list / vp_abs_norm_y_list > 3 ), 1, 0)
    is_diagonal_vp_candidate =    np.where((vp_abs_norm_x_list / vp_abs_norm_y_list <= 3) & (vp_abs_norm_y_list / vp_abs_norm_x_list <= 3) , 1, 0)
    is_vertical_vp_candidate =    np.where((vp_abs_norm_y_list / vp_abs_norm_x_list > 7) & (vp_abs_norm_y_list > 1.5) ,  1, 0) # 8.13度くらいの傾きなら補正する。それより大きければ無視。

    is_far_l_horizontal_vp_candidate = np.where((vp_abs_norm_x_list / vp_abs_norm_y_list > 2.5) & (vp_norm_x_list < -20), 1, 0)
    is_far_r_horizontal_vp_candidate = np.where((vp_abs_norm_x_list / vp_abs_norm_y_list > 2.5) & (vp_norm_x_list > 20), 1, 0)

    if sum(is_horizontal_vp_candidate) == 0: # どうしようも無いのではるか右側遠方に飛ばす
        vp_horizontal_index = 3
        vp_h = np.array([INF, 0, 1])
    else:
        vp_horizontal_index = np.argmax(vp_abs_norm_x_list * is_horizontal_vp_candidate)
        vp_h = vp_list[vp_horizontal_index]

    if sum(is_vertical_vp_candidate) == 0: # どうしようも無いのではるか下側遠方に飛ばす
        vp_vertical_index = 3
        vp_v = np.array([0, INF, 1])
    else:
        vp_vertical_index = np.argmax(vp_abs_norm_y_list * is_vertical_vp_candidate)
        vp_v = vp_list[vp_vertical_index]

    vp_other_index = [i for i in range(3) if (i != vp_horizontal_index and i != vp_vertical_index)][0]
    vp_other = vp_list[vp_other_index]

    # 縦消失点と横消失点が強制変更の影響で一致してしまった場合への対応（普通はありえない）
    if (vp_v/vp_v[2] == vp_h/vp_h[2]).all():
        vp_horizontal_index = 3
        vp_h= np.array([INF, 0, 1])
        vp_vertical_index = 3
        vp_v = np.array([0, INF, 1])


    # 遠近変換方向に合わせて消失点の推測値を修正
    limit_v = 10
    limit_h = 15

    if is_limited and np.abs(vp_v[1]/vp_v[2] - img_h/2) < limit_v * img_h/2:
        vp_v[1] =  (limit_v * img_h/2 * np.sign(vp_v[1]/vp_v[2] - img_h/2) + img_h/2) * vp_v[2] # 上の条件式を式変形した結果

    if is_limited and np.abs(vp_h[0]/vp_h[2] - img_w/2) < limit_h * img_w/2:
        vp_h[0] =  (limit_h * img_w/2 * np.sign(vp_h[0]/vp_h[2] - img_w/2) + img_w/2) * vp_h[2] # 上の条件式を式変形した結果

    if align == 'vertical':
        vp_h[0] = INF*np.sign(vp_h[0])
        vp_h[1] = 0

    elif align == 'horizontal':
        vp_v[1] = INF*np.sign(vp_v[1])
        vp_v[0] = 0

    elif align == 'auto':
        # 左遠方と下方に消失点があり，右遠方には消失点が無い場合 → 1点透視
        if sum(is_far_l_horizontal_vp_candidate) >= 1 and sum(is_far_r_horizontal_vp_candidate) == 0:
            vp_horizontal_index = np.argmax(vp_abs_norm_x_list * is_far_l_horizontal_vp_candidate)
            vp_h = vp_list[vp_horizontal_index]
            if is_limited and np.abs(vp_h[0]/vp_h[2] - img_w/2) < limit_h * img_w/2:
                vp_h[0] =  (limit_h * img_w/2 * np.sign(vp_h[0]/vp_h[2] - img_w/2) + img_w/2) * vp_h[2] # 上の条件式を式変形した結果

        # 右遠方と下方に消失点があり，左遠方には消失点が無い場合 → 1点透視
        elif sum(is_far_l_horizontal_vp_candidate) == 0 and sum(is_far_r_horizontal_vp_candidate) >= 1:
            vp_horizontal_index = np.argmax(vp_abs_norm_x_list * is_far_r_horizontal_vp_candidate)
            vp_h = vp_list[vp_horizontal_index]
            if is_limited and np.abs(vp_h[0]/vp_h[2] - img_w/2) < limit_h * img_w/2:
                vp_h[0] =  (limit_h * img_w/2 * np.sign(vp_h[0]/vp_h[2] - img_w/2) + img_w/2) * vp_h[2] # 上の条件式を式変形した結果

        # 上記以外 → 2点透視
        else:
            vp_h[0] = INF*np.sign(vp_h[0])
            vp_h[1] = 0

    return vp_h, vp_v, vp_other


def crop_warped_image(img, warped_img, transform_matrix):
    TODO:autoもしくはhorizontalのときのcropアルゴリズムを追加
    img_h, img_w, _ = img.shape

    img_corners = np.array([[0, 0, img_w, img_w],
                            [0, img_h, 0, img_h],
                            [1, 1, 1, 1]])

    warped_img_corners = np.dot(transform_matrix, img_corners)
    warped_img_corners = warped_img_corners[:2] / warped_img_corners[2]

    ul_warped = warped_img_corners[:,0]
    dl_warped = warped_img_corners[:,1]
    ur_warped = warped_img_corners[:,2]
    dr_warped = warped_img_corners[:,3]

    left_crop  = int(round(max(ul_warped[0], dl_warped[0])))
    right_crop = int(round(min(ur_warped[0], dr_warped[0])))
    up_crop    = int(round(max(ul_warped[1], ur_warped[1])))
    down_crop  = int(round(min(dl_warped[1], dr_warped[1])))

    cropped_image = np.zeros((down_crop-up_crop, right_crop-left_crop, 3), np.uint8)
    cropped_image = warped_img[up_crop:down_crop, left_crop:right_crop, :]

    return cropped_image

def padding_cropped_image(img, cropped_img):
    img_h, img_w, _ = img.shape
    cropped_img_h, cropped_img_w, _ = cropped_img.shape

    if cropped_img_h > img_h and cropped_img_w == img_w:
        padded_cropped_img_w = int(cropped_img_w * (cropped_img_h/img_h))
        padding_w = (padded_cropped_img_w - cropped_img_w)//2
        padded_cropped_img_h = int(cropped_img_h)
        padded_cropped_img = np.full((padded_cropped_img_h, padded_cropped_img_w, 3), 255, dtype= np.uint8)
        padded_cropped_img[:,padding_w:padding_w+cropped_img_w,:] = cropped_img

        return padded_cropped_img

    return cropped_img


def rectify_image(image, num_ransac_iter=500, reestimate=False, limit_v = 10, limit_h=15, align='vertical', is_resized=False):
    """Rectified image with vanishing point computed using ransac.
    Parameters
    ----------
    image: ndarray
        Image which has to be rectified.
    clip_factor: float, optional
        Proportion of image in multiples of image size to be retained if gone
        out of bounds after homography.
    algorithm: one of {'3-line', 'independent'}
        independent ransac algorithm finds the orthogonal vanishing points by
        applying ransac twice.
        3-line algorithm finds the orthogonal vanishing points together, but
        assumes knowledge of focal length.
    reestimate: bool
        If ransac results are to be reestimated using least squares with
        inlers. Turn this off if getting bad results.
    Returns
    -------
    warped_img: ndarray
        Rectified image.
    """
   # if type(image) is not np.ndarray:w
      #  image = cv2.imread(image)

    decoded_data = base64.b64decode(image)
    np_data = np.fromstring(decoded_data, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

    if align != 'horizontal' and align != 'vertical' and align != 'auto':
        print('align shoud be auto or horizontal or vertical')

    vp1, vp2, vp3 = find_vanishing_points(image, num_ransac_iter, reestimate)

    INF = 10**15
    img_h, img_w, _ = image.shape
    vp_h, vp_v, vp_other = classify_vanishing_points(img_h, img_w, [vp1, vp2, vp3], INF, limit_v, limit_h, align)
    # print('縦方向遠方の消失点の座標（同次座標表示）', vp_v/vp_v[2])
    # print('横方向遠方の消失点の座標（同次座標表示）', vp_h/vp_h[2])


    # Compute the homography and warp
    warped_img, warp_homography = compute_homography_and_warp(image, vp_h, vp_v)

    output_img = crop_warped_image(image, warped_img, warp_homography)
    # output_img = padding_cropped_image(image, output_img) #とりあえず今はPaddingしない

    if is_resized:
        output_img = cv2.resize(output_img, (int(image.shape[1]),int(image.shape[0])), interpolation=cv2.INTER_LANCZOS4)

    return output_img.BytesIO()

def test():
    return "string"