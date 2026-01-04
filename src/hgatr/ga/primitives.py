import torch
import torch.nn.functional as F
import numpy as np

## The blade is a multivector <x>_k operator. Applied to a multivector x,
## it returns x with only the components of grade k.
def blade_operator(mv_dimension, grade_components, ga_permutation):
    """
    Generates a blade operator matrix for the geometric algebra

    Args:
        None

    Returns:
        (torch.Tensor): Blade operator matrix.
    """

    blade_shape = (mv_dimension, mv_dimension)

    coordinates = []
    start = 0
    for length in grade_components:
        coordinates.append(list(range(start, start + length)))
        start += length

    coord_permutations = ga_permutation
    blade_mask = []

    w_dimension = len(grade_components)
    for k_grade in range(w_dimension):
        w_blade = torch.zeros(blade_shape)
        for coordinate in coordinates[k_grade]:
            w_blade[coordinate, coordinate] = 1.0
        blade_mask.append(w_blade.unsqueeze(0))

    v_dimension = len(grade_components) - 1
    for k_grade in range(v_dimension):
        v_blade = torch.zeros(blade_shape)
        for coord_to,coord_from in coord_permutations[k_grade]:
            if coord_from < mv_dimension and coord_to < mv_dimension:
                v_blade[coord_from, coord_to] = 1.0
            else:
                print(f"Warning: Indici fuori limiti: coord_from={coord_from}, coord_to={coord_to}")
        blade_mask.append(v_blade.unsqueeze(0))

    blade_operator = torch.cat(blade_mask,dim = 0)

    return blade_operator


## The reverse is an operator to be applied to a multivector that changes the signs of its bivectors and trivectors.
## In applications, it can be reassimilated to a transposition.
def get_coordinates_range(grade_components):
    """
    Get the ranges of coordinates for each grade based on the configuration in `global_var`.

    Args:
        None

    Returns:
        (List[List[int]]): List of coordinate ranges for each grade.
    """
    coordinates_range = []

    for grade in range(len(grade_components)):
        start_idx = sum(grade_components[:grade])
        end_idx = sum(grade_components[:grade + 1]) - 1
        coordinate_range = [start_idx,end_idx]
        coordinates_range.append(coordinate_range)

    return coordinates_range


def reverse_operator(ga_dimension, device = "cpu"):
    """
    Generate a reverse operator for the given multivector configuration.

    Args:
        None

    Returns:
        (torch.Tensor): Reverse operator for the multivector space.
    """

    reverse_operator = torch.ones(ga_dimension)
    *_, bivector_range, trivector_range, _ = get_coordinates_range()

    reverse_range = list(
        range(
            bivector_range[0],
            trivector_range[-1] + 1
         )
     )

    reverse_operator[reverse_range] = -1

    return reverse_operator.to(device)


## The reverse x^* is an operator that generalize the blade.
## It changes the signs of the even mutlivector components.
def dual_operators(ga_dimension, device="cpu"):
    """
    Constructs dual operators, including the indices for sign flipping and the sign values.

    Args:
        None

    Returns:
        (Tuple[List[int], torch.Tensor]): Tuple containing a list of indices to mask to perfom dual and
                                          the vector to perfom it. This masked approach make faster the
                                          dual computation when applied to a multivector
    """

    coords_range = get_coordinates_range()

    dual_sign_idxs = [2,4,6,9,12,14]
    dual_signs = [
        1 if i not in dual_sign_idxs else -1
        for i in range(ga_dimension)
    ][::-1]
    dual_signs = torch.tensor(dual_signs)

    dual_flip = list(
        range(
            coords_range[-1][0],coords_range[0][0]-1,-1
         )
     )

    return dual_flip, dual_signs.to(device)


## GEOMETRIC PRODUCT
## The geometric product xy is a particular multiplication that has two multivectors as arguments.
## This product respects some properties:
## - The product of the same argument must be equal to its norm squared: vv = ||v||^2.
## - The product between bases is antisymmetric: e_ie_j = -e_je_i ∀i =/ j.
## - The product of the space basis squared is one: e_i^2 = 1 ∀i =/ 0.
## - The product of the projective basis squared is zero: e_0^2= 0.
##
## To realize this particular product, Einstein's summation convention is very convenient.
## This convention provides a concise way to express complex multidimensional tensor operations.
##
## In order to obtain the properties correctly, after some unsuccessful attempts or otherwise
## not accurate and consistent for the desired results, we have exploited a guiding matrix proposed
## by [1,3] to correctly and efficiently realize the product operations (geometric and non-geometric)
## of the geometric algebra in question.

def get_guidance_matrix():
    """
    Fetches and loads the guidance matrix from a specified URL,
    saving it locally if not already present.

    Returns:
        (torch.Tensor): A tensor representing the guidance matrix.
    """
    guidance_matrix = product_basis = torch.load("guidance_matrix.pt", weights_only=True)
    guidance_matrix = guidance_matrix.to(torch.float32)
    guidance_matrix = guidance_matrix.to_dense()

    return guidance_matrix

def geometric_product(x,y, device="cpu"):
    """
    Computes the geometric product of two multivectors using the guidance matrix.

    Args:
        x (torch.Tensor): the first multivector
        y (torch.Tensor): the second multivector

    Returns:
        (torch.Tensor): multivector result of the geometric product
    """

    guidance_matrix = get_guidance_matrix().to(device)

    geom_prod = torch.einsum(
        "i j k, ... j, ... k -> ... i",
        guidance_matrix,
        x,
        y
     )

    return geom_prod


## Inner product
## The inner product <x,y> is a generalized version of the well-known scalar product,
## applied to multivectors. Conceptually, it indicates the similarity between two multivectors.
## In the practical context, to compute the inner product, we exploit its form <x,y> = <\tilde{x}y>_0,
## thus computing the blade of degree 0 related to the geometric product between the inverse of the first operand
## and the second operand.
## This type of approach is correct but may become very slow for tensors with enough elements.
## Thus [1], as done for the dual operator, proposes a mask approach that isolates only the components that
## will actually be affected by the geometric product, thus achieving the same result but in less time.
def inner_product(x,y):
    """
    Computes the inner product of two multivectors.

    Args:
        x (torch.Tensor): the first multivector
        y (torch.Tensor): the second multivector

    Returns:
        (torch.Tensor): result of the inner product
    """

    reverse_x = reverse_operator() * x
    geom_x_y = geometric_product(reverse_x,y)
    inner_product = geom_x_y[..., [0]]

    return inner_product

def faster_inner_product(x,y, device="cpu"):
    """
    Computes the inner product of two multivectors using a faster method.

    Args:
        x (torch.Tensor): the first multivector
        y (torch.Tensor): the second multivector

    Returns:
        (torch.Tensor): scalar result of the inner product
    """
    guidance_matrix = get_guidance_matrix().to(device)
    reverse_op = reverse_operator()

    inner_product_mask = (
        torch.diag(guidance_matrix[0]) * reverse_op
    ).bool()

    x = x[..., inner_product_mask]
    y = y[..., inner_product_mask]

    # questo è un "geom prod2 modificato"
    inner_product = torch.einsum(
        "... i, ... i -> ...", x, y
    ).unsqueeze(-1)


    return inner_product


### OUTER PRODUCT
## The outer product x \wedge y represents another particular product of algebraic geometry,
## conceptually close to the well-known vector product. This product follows the thread of the geometric product,
## but with a slightly different product guide, again proposed and implemented by [1].

def get_outer_matrix():
    """
    Fetches and loads the outer product guidance matrix
    from a specified URL,  saving it locally if not already present.

    Returns:
        (torch.Tensor): A tensor representing the outer guidance matrix.
    """

    #if not os.path.exists("outer_guidance.pt"):
    #   matrix_url = global_var['prefix'] + global_var['outer_id']
    #   matrix_file = wget.download(matrix_url)

    outer_matrix = torch.load("outer_guidance.pt")
    outer_matrix = outer_matrix.to(torch.float32)
    outer_matrix = outer_matrix.to_dense()

    return outer_matrix

def outer_product(x,y, device="cpu"):
    """
    Computes the outer product of two multivectors using the outer guidance matrix.

    Args:
        x (torch.Tensor): the first multivector
        y (torch.Tensor): the second multivector

    Returns:
        (torch.Tensor): multivector result of the outer product
    """

    outer_matrix = get_outer_matrix().to(device)

    outputs = torch.einsum("i j k, ... j, ... k -> ... i", outer_matrix, x, y)
    return outputs

## Join
## The join x,y = (x^* \wedge y^*)^* is an operation that, conceptually is like placing an object of geometric algebra between the two elements x and y,
## as a "line" connecting them. In project algebraic geometry, however, the intrinsic nature of the join is nonequivariant.
## The authors of [1] propose a solution by multiplying the result of the join by a reference, i.e., the average of the first multivector entering the GATr.
## This technique is demonstrated in the paper [1] and discussed in this video(https://www.youtube.com/watch?v=YbTSgnzg26Q&t=2505s).
def join(x,y,ref, ga_dimension):
    """
    Computes the join of two multivectors relative to a reference multivector.

    Args:
        x (torch.Tensor): the first multivector
        y (torch.Tensor): the second multivector
        ref (torch.Tensor): the reference multivector, mean over the batch
                            and channels of the entering GATr multivector

    Returns:
        (torch.Tensor): multivector result of the join operation
    """

    dual_flip,dual_signs = dual_operators()
    dual_x = dual_signs * x[...,dual_flip]
    dual_y = dual_signs * y[...,dual_flip]

    outer_prod = outer_product(dual_x, dual_y)
    classic_join = dual_signs * outer_prod[...,dual_flip]
    equi_join = ref[..., [ga_dimension - 1]] * classic_join

    return equi_join


## Grade involution
## The grade involution \hat{x} is an operator that slightly modifies the blade operator.
## In fact, it changes the signs of the odd mutlivector components.
def grade_involution(mv, ga_dimension, device="cpu"):
    """
    Applies the grade involution to a multivector, flipping the signs of odd-graded components.

    Args:
        mv (torch.Tensor): the input multivector

    Returns:
        (torch.Tensor): multivector result of the grade involution
        (list): indices of flipped signs during the involution
    """


    odd_grades = get_coordinates_range()[1::2]
    flip_signs = []

    for grade_range in odd_grades:
        flip_signs += list(range(grade_range[0],grade_range[1] + 1))

    involution_signs = [-1 if i in flip_signs else 1 for i in range(ga_dimension)][::-1]
    involution_signs = torch.tensor(involution_signs).to(device)

    involuted_mv = involution_signs * mv

    return involuted_mv, flip_signs


## Sandwich product
## The sandwich product r_u(x) allows the operator u to be applied to the multivector x.
## The application of the latter is done by a geometric product between u, x and the inverse operator u^{-1}.
## In particular, if the operator has elements of odd degree, the multivector being applied must undergo
## grade involution, before performing the above products.
def sandwich_product(mv,u,inv_u):
    """
    Computes the sandwich product of a multivector and a matrix.

    Args:
        mv (torch.Tensor): the input multivector
        u (torch.Tensor): the matrix for the sandwich product

    Returns:
        (torch.Tensor): a torch tensor representing the result
                        of the sandwich product
    """
    first_geom_product = geometric_product(u,mv)
    output = geometric_product(first_geom_product,inv_u)

    return output


## Equivariant check
## The particularity of the neural network of the project, the GATr proposed by [1] is its ability to be
## equivariant with respect to the operators of E(3), the rigid transformations of 3D space: rotations,
## translations and reflections. This particularity is an important element to achieve, as it differs
## from other Transformer-type architectures and will be able to be an important term of comparison.
## Indeed, as noted in the EDA section, the binary classification task, could prove to be quite simple,
## especially for very powerful architectures such as those that will be presented. So the peculiarity
## of having an equivariant approach would certainly be a point in favor of the GATr architecture.
## To check that indeed this property is respected, the equation to check is as follows.

def difference_distance(first_side,second_side):
    """
    Computes the difference distance between two sides represented by tensors.

    Args:
        first_side (torch.Tensor): the first side represented as a tensor
        second_side (torch.Tensor): the second side represented as a tensor

    Returns:
        (float): the Euclidean norm of the difference between the two sides
    """

    first_side = first_side.cpu().detach().numpy()
    second_side = second_side.cpu().detach().numpy()
    return np.linalg.norm(first_side - second_side)


def equivariant_check(mv,operators,layer,ref=None, device="cpu"):
    """
    Checks the equivariance of a GATr layer with respect to a multivector.

    Args:
        mv (torch.Tensor): the input multivector
        operators (List): list containing the direct and inverse operators
        layer (torch.nn.Module): the GATr layer
        ref (torch.Tensor, optional): reference multivector for join

    Returns:
        (List): contains the two equivariant equation sides and normalized difference
                distance between them to assess equivariance
    """

    batch, items, channels, _ = mv.shape
    denominator = batch * items * channels
    equicheck_in = mv.to(device)
    operator = operators[0].to(device)
    inv_operator = operators[1].to(device)
    involuted_mv, flipped_signs = grade_involution(equicheck_in)
    layer = layer.to(device)

    if torch.any(operator[...,flipped_signs] != 0):
        equicheck_in = involuted_mv.to(device)

    if ref is not None:
        first_side = sandwich_product(
            mv = layer(equicheck_in,ref),
            u = operator,
            inv_u = inv_operator
        )
        second_side = layer(
            sandwich_product(
                mv = equicheck_in,
                u = operator,
                inv_u = inv_operator
                ),
            ref
        )
    else:
        first_side = sandwich_product(
            mv = layer(equicheck_in),
            u = operator,
            inv_u = inv_operator
        )
        second_side = layer(
            sandwich_product(
                mv = equicheck_in,
                u = operator,
                inv_u = inv_operator
            )
        )

    output = difference_distance(first_side,second_side)
    output /= denominator

    return output